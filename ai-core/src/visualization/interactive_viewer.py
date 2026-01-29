"""
InteractiveAnalysisViewer: Interactive GUI for analyzing lifting metrics.

Features:
- 3-panel layout: Left graph | Center video | Right graph
- Dropdown selectors for choosing which metric to display
- Video playback controls (play/pause/stop, slow-motion 0.25x)
- Video trimming with markers to select a portion
- Time indicator on graphs synced with video position
- Graphs update to show only the trimmed portion
- Trajectory X-Y plot with optional velocity colormap
- Adapts to vertical or horizontal video format
- White background with margins between elements
- Frame number and time scale clearly visible

Requires: PyQt5, matplotlib, opencv-python, pandas
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSlider, QFrame, QSizePolicy,
    QCheckBox, QSpinBox, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QPalette

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Available graphs configuration
GRAPH_OPTIONS = {
    "Altura (m)": {
        "column": "height_m",
        "ylabel": "Altura (m)",
        "color": "#2ecc71",
        "title": "Altura del Disco vs Tiempo",
        "type": "time_series"
    },
    "Velocidad (m/s)": {
        "column": "speed_m_s",
        "ylabel": "Velocidad (m/s)",
        "color": "#3498db",
        "title": "Velocidad del Disco vs Tiempo",
        "type": "time_series"
    },
    "Velocidad Y (m/s)": {
        "column": "vy_m_s",
        "ylabel": "Vel. Vertical (m/s)",
        "color": "#9b59b6",
        "title": "Velocidad Vertical vs Tiempo",
        "type": "time_series"
    },
    "Aceleración (m/s²)": {
        "column": "accel_m_s2",
        "ylabel": "Aceleración (m/s²)",
        "color": "#e74c3c",
        "title": "Aceleración vs Tiempo",
        "type": "time_series"
    },
    "Energía Cinética (J)": {
        "column": "kinetic_energy_j",
        "ylabel": "Energía (J)",
        "color": "#9b59b6",
        "title": "Energía Cinética vs Tiempo",
        "type": "time_series"
    },
    "Energía Potencial (J)": {
        "column": "potential_energy_j",
        "ylabel": "Energía (J)",
        "color": "#f39c12",
        "title": "Energía Potencial vs Tiempo",
        "type": "time_series"
    },
    "Energía Total (J)": {
        "column": "total_energy_j",
        "ylabel": "Energía (J)",
        "color": "#1abc9c",
        "title": "Energía Mecánica Total vs Tiempo",
        "type": "time_series"
    },
    "Potencia (W)": {
        "column": "power_w",
        "ylabel": "Potencia (W)",
        "color": "#e91e63",
        "title": "Potencia Instantánea vs Tiempo",
        "type": "time_series"
    },
    "Posición X (m)": {
        "column": "x_m",
        "ylabel": "Posición X (m)",
        "color": "#00bcd4",
        "title": "Posición Horizontal vs Tiempo",
        "type": "time_series"
    },
    "Posición Y (m)": {
        "column": "y_m",
        "ylabel": "Posición Y (m)",
        "color": "#ff5722",
        "title": "Posición Vertical vs Tiempo",
        "type": "time_series"
    },
    "📍 Trayectoria X-Y": {
        "type": "trajectory",
        "title": "Trayectoria del Disco",
        "color": "#2c3e50"
    },
}

# Available colormaps for trajectory
COLORMAPS = [
    "plasma",      # Default - good for velocity (purple to yellow)
    "viridis",     # Blue to yellow
    "inferno",     # Black to yellow through red
    "magma",       # Black to white through purple
    "coolwarm",    # Blue to red (good for showing direction changes)
    "jet",         # Classic rainbow
    "turbo",       # Improved rainbow
    "RdYlGn",      # Red to green
    "Spectral",    # Colorful spectrum
]


# Common stylesheet for dark text on white background
DROPDOWN_STYLE = """
    QComboBox {
        background-color: #f5f5f5;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 5px 10px;
        min-height: 25px;
        font-size: 12px;
    }
    QComboBox:hover {
        border-color: #3498db;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox::down-arrow {
        width: 12px;
        height: 12px;
    }
    QComboBox QAbstractItemView {
        background-color: white;
        color: #333333;
        selection-background-color: #3498db;
        selection-color: white;
        border: 1px solid #cccccc;
    }
"""

DROPDOWN_SMALL_STYLE = """
    QComboBox {
        background-color: #f5f5f5;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 3px;
        padding: 3px 8px;
        min-height: 20px;
        font-size: 10px;
    }
    QComboBox:hover {
        border-color: #3498db;
    }
    QComboBox QAbstractItemView {
        background-color: white;
        color: #333333;
        selection-background-color: #3498db;
        selection-color: white;
        font-size: 10px;
    }
"""

CHECKBOX_STYLE = """
    QCheckBox {
        color: #333333;
        font-size: 10px;
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
    }
"""

SPINBOX_STYLE = """
    QSpinBox {
        background-color: #f5f5f5;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 3px;
        padding: 2px 5px;
        min-height: 20px;
        font-size: 10px;
    }
"""

BUTTON_STYLE = """
    QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 15px;
        font-size: 12px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #2980b9;
    }
    QPushButton:pressed {
        background-color: #1f618d;
    }
    QPushButton:disabled {
        background-color: #bdc3c7;
    }
"""

BUTTON_ACTIVE_STYLE = """
    QPushButton {
        background-color: #27ae60;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 15px;
        font-size: 12px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #219a52;
    }
"""

BUTTON_DANGER_STYLE = """
    QPushButton {
        background-color: #e74c3c;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 15px;
        font-size: 12px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #c0392b;
    }
"""


class TrimSlider(QWidget):
    """Custom slider with trim markers for selecting a video range."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.minimum = 0
        self.maximum = 100
        self.value = 0
        self.trim_start = 0
        self.trim_end = 100
        self.is_trimmed = False
        
        # Dragging state
        self.dragging = None  # None, 'start', 'end', 'value'
        
        self.setMinimumHeight(50)
        self.setMouseTracking(True)
        
        # Callbacks
        self.on_value_changed = None
        self.on_trim_changed = None
    
    def setRange(self, min_val: int, max_val: int):
        self.minimum = min_val
        self.maximum = max_val
        self.trim_start = min_val
        self.trim_end = max_val
        self.update()
    
    def setValue(self, val: int):
        self.value = max(self.minimum, min(val, self.maximum))
        self.update()
    
    def setTrimRange(self, start: int, end: int):
        self.trim_start = max(self.minimum, min(start, self.maximum))
        self.trim_end = max(self.minimum, min(end, self.maximum))
        self.is_trimmed = (self.trim_start > self.minimum or self.trim_end < self.maximum)
        self.update()
    
    def resetTrim(self):
        self.trim_start = self.minimum
        self.trim_end = self.maximum
        self.is_trimmed = False
        self.update()
        if self.on_trim_changed:
            self.on_trim_changed(self.trim_start, self.trim_end)
    
    def _value_to_x(self, val: int) -> int:
        """Convert value to x coordinate."""
        margin = 15
        usable_width = self.width() - 2 * margin
        if self.maximum == self.minimum:
            return margin
        return margin + int((val - self.minimum) / (self.maximum - self.minimum) * usable_width)
    
    def _x_to_value(self, x: int) -> int:
        """Convert x coordinate to value."""
        margin = 15
        usable_width = self.width() - 2 * margin
        if usable_width <= 0:
            return self.minimum
        val = self.minimum + (x - margin) / usable_width * (self.maximum - self.minimum)
        return max(self.minimum, min(int(val), self.maximum))
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        margin = 15
        track_height = 8
        track_y = self.height() // 2 - track_height // 2
        
        # Draw background track
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#e0e0e0"))
        painter.drawRoundedRect(margin, track_y, self.width() - 2*margin, track_height, 4, 4)
        
        # Draw trim region (highlighted)
        trim_start_x = self._value_to_x(self.trim_start)
        trim_end_x = self._value_to_x(self.trim_end)
        
        if self.is_trimmed:
            # Dimmed regions outside trim
            painter.setBrush(QColor(200, 200, 200, 150))
            painter.drawRect(margin, track_y, trim_start_x - margin, track_height)
            painter.drawRect(trim_end_x, track_y, self.width() - margin - trim_end_x, track_height)
        
        # Active trim region
        painter.setBrush(QColor("#3498db"))
        painter.drawRoundedRect(trim_start_x, track_y, trim_end_x - trim_start_x, track_height, 4, 4)
        
        # Draw trim markers
        marker_height = 25
        marker_width = 8
        marker_y = self.height() // 2 - marker_height // 2
        
        # Start marker (green)
        painter.setBrush(QColor("#27ae60"))
        painter.setPen(QPen(QColor("#1e8449"), 2))
        painter.drawRoundedRect(trim_start_x - marker_width//2, marker_y, marker_width, marker_height, 2, 2)
        
        # End marker (red)
        painter.setBrush(QColor("#e74c3c"))
        painter.setPen(QPen(QColor("#c0392b"), 2))
        painter.drawRoundedRect(trim_end_x - marker_width//2, marker_y, marker_width, marker_height, 2, 2)
        
        # Draw current position marker
        pos_x = self._value_to_x(self.value)
        painter.setBrush(QColor("#2c3e50"))
        painter.setPen(QPen(QColor("#1a252f"), 2))
        painter.drawEllipse(pos_x - 8, self.height()//2 - 8, 16, 16)
        
        # Draw position line
        painter.setPen(QPen(QColor("#2c3e50"), 2))
        painter.drawLine(pos_x, 5, pos_x, self.height() - 5)
    
    def mousePressEvent(self, event):
        x = event.x()
        pos_x = self._value_to_x(self.value)
        start_x = self._value_to_x(self.trim_start)
        end_x = self._value_to_x(self.trim_end)
        
        # Check which element was clicked (with tolerance)
        tolerance = 15
        
        if abs(x - start_x) < tolerance:
            self.dragging = 'start'
        elif abs(x - end_x) < tolerance:
            self.dragging = 'end'
        else:
            self.dragging = 'value'
            new_val = self._x_to_value(x)
            self.value = new_val
            self.update()
            if self.on_value_changed:
                self.on_value_changed(self.value)
    
    def mouseMoveEvent(self, event):
        if self.dragging is None:
            return
        
        x = event.x()
        new_val = self._x_to_value(x)
        
        if self.dragging == 'start':
            self.trim_start = min(new_val, self.trim_end - 1)
            self.is_trimmed = True
            self.update()
            if self.on_trim_changed:
                self.on_trim_changed(self.trim_start, self.trim_end)
        elif self.dragging == 'end':
            self.trim_end = max(new_val, self.trim_start + 1)
            self.is_trimmed = True
            self.update()
            if self.on_trim_changed:
                self.on_trim_changed(self.trim_start, self.trim_end)
        elif self.dragging == 'value':
            self.value = new_val
            self.update()
            if self.on_value_changed:
                self.on_value_changed(self.value)
    
    def mouseReleaseEvent(self, event):
        self.dragging = None


class GraphPanel(QWidget):
    """A panel containing a dropdown selector and a matplotlib graph."""
    
    def __init__(self, metrics_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.full_metrics_df = metrics_df
        self.metrics_df = metrics_df  # Current view (may be trimmed)
        self.current_time = 0.0
        self.time_line = None
        self.current_position_marker = None
        
        # Trajectory options
        self.show_velocity_color = True
        self.line_thickness = 3
        self.colormap_name = "plasma"
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Dropdown selector with fixed styling
        self.selector = QComboBox()
        self.selector.addItems(GRAPH_OPTIONS.keys())
        self.selector.setFont(QFont("Arial", 11))
        self.selector.setStyleSheet(DROPDOWN_STYLE)
        self.selector.currentTextChanged.connect(self.on_selection_changed)
        layout.addWidget(self.selector)
        
        # Trajectory options panel (hidden by default)
        self.trajectory_options = QWidget()
        traj_layout = QHBoxLayout(self.trajectory_options)
        traj_layout.setContentsMargins(5, 5, 5, 5)
        traj_layout.setSpacing(10)
        
        # Velocity color checkbox
        self.velocity_checkbox = QCheckBox("Color por velocidad")
        self.velocity_checkbox.setChecked(True)
        self.velocity_checkbox.setStyleSheet(CHECKBOX_STYLE)
        self.velocity_checkbox.stateChanged.connect(self.on_velocity_toggle)
        traj_layout.addWidget(self.velocity_checkbox)
        
        # Line thickness
        thickness_label = QLabel("Grosor:")
        thickness_label.setStyleSheet("color: #333; font-size: 10px;")
        traj_layout.addWidget(thickness_label)
        
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 10)
        self.thickness_spin.setValue(3)
        self.thickness_spin.setStyleSheet(SPINBOX_STYLE)
        self.thickness_spin.valueChanged.connect(self.on_thickness_changed)
        traj_layout.addWidget(self.thickness_spin)
        
        # Colormap selector
        cmap_label = QLabel("Colormap:")
        cmap_label.setStyleSheet("color: #333; font-size: 10px;")
        traj_layout.addWidget(cmap_label)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(COLORMAPS)
        self.colormap_combo.setStyleSheet(DROPDOWN_SMALL_STYLE)
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        traj_layout.addWidget(self.colormap_combo)
        
        traj_layout.addStretch()
        
        self.trajectory_options.hide()
        layout.addWidget(self.trajectory_options)
        
        # Matplotlib canvas
        self.figure = Figure(figsize=(5, 4), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        
        # Initial plot
        self.plot_graph(list(GRAPH_OPTIONS.keys())[0])
    
    def set_trim_range(self, start_frame: int, end_frame: int):
        """Update the displayed data to show only the trimmed range."""
        mask = (self.full_metrics_df['frame_idx'] >= start_frame) & \
               (self.full_metrics_df['frame_idx'] <= end_frame)
        self.metrics_df = self.full_metrics_df[mask].copy()
        
        # Replot current graph
        current_selection = self.selector.currentText()
        self.plot_graph(current_selection)
    
    def reset_trim(self):
        """Reset to show full data."""
        self.metrics_df = self.full_metrics_df
        current_selection = self.selector.currentText()
        self.plot_graph(current_selection)
    
    def on_selection_changed(self, text: str):
        config = GRAPH_OPTIONS.get(text, {})
        
        # Show/hide trajectory options
        if config.get("type") == "trajectory":
            self.trajectory_options.show()
        else:
            self.trajectory_options.hide()
        
        self.plot_graph(text)
    
    def on_velocity_toggle(self, state):
        self.show_velocity_color = (state == Qt.Checked)
        current_selection = self.selector.currentText()
        if GRAPH_OPTIONS.get(current_selection, {}).get("type") == "trajectory":
            self.plot_graph(current_selection)
    
    def on_thickness_changed(self, value):
        self.line_thickness = value
        current_selection = self.selector.currentText()
        if GRAPH_OPTIONS.get(current_selection, {}).get("type") == "trajectory":
            self.plot_graph(current_selection)
    
    def on_colormap_changed(self, name):
        self.colormap_name = name
        current_selection = self.selector.currentText()
        if GRAPH_OPTIONS.get(current_selection, {}).get("type") == "trajectory":
            self.plot_graph(current_selection)
    
    def plot_graph(self, graph_name: str):
        """Plot the selected graph."""
        self.figure.clear()
        
        config = GRAPH_OPTIONS.get(graph_name, {})
        graph_type = config.get("type", "time_series")
        
        if graph_type == "trajectory":
            self._plot_trajectory(config)
        else:
            self._plot_time_series(graph_name, config)
    
    def _plot_time_series(self, graph_name: str, config: dict):
        """Plot a standard time series graph."""
        ax = self.figure.add_subplot(111)
        
        column = config.get("column", "height_m")
        ylabel = config.get("ylabel", "Value")
        color = config.get("color", "#3498db")
        title = config.get("title", graph_name)
        
        if self.metrics_df.empty or column not in self.metrics_df.columns:
            ax.text(0.5, 0.5, f"No data available", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='#666')
            self.canvas.draw()
            return
        
        time = self.metrics_df['time_s'].values
        values = self.metrics_df[column].values
        frames = self.metrics_df['frame_idx'].values
        
        # Plot data
        ax.plot(time, values, color=color, linewidth=2, label=ylabel)
        ax.fill_between(time, 0, values, color=color, alpha=0.2)
        
        # Labels and title
        ax.set_xlabel('Tiempo (s)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add secondary x-axis for frame numbers
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        n_ticks = min(6, len(frames))
        if n_ticks > 1:
            frame_ticks = np.linspace(frames[0], frames[-1], n_ticks).astype(int)
            time_ticks = np.interp(frame_ticks, frames, time)
            ax2.set_xticks(time_ticks)
            ax2.set_xticklabels([f'F{f}' for f in frame_ticks], fontsize=8)
        ax2.set_xlabel('Frame', fontsize=9)
        
        # Time indicator line (vertical)
        if len(time) > 0:
            self.time_line = ax.axvline(x=self.current_time, color='red', 
                                        linewidth=2, linestyle='--', alpha=0.8)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _plot_trajectory(self, config: dict):
        """Plot the X-Y trajectory, optionally colored by velocity."""
        ax = self.figure.add_subplot(111)
        
        if self.metrics_df.empty or 'x_m' not in self.metrics_df.columns:
            ax.text(0.5, 0.5, "No trajectory data available", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='#666')
            self.canvas.draw()
            return
        
        x = self.metrics_df['x_m'].values
        y = self.metrics_df['height_m'].values  # Use height for Y (more intuitive)
        
        title = config.get("title", "Trayectoria del Disco")
        
        if self.show_velocity_color and 'speed_m_s' in self.metrics_df.columns:
            # Color by velocity
            speed = self.metrics_df['speed_m_s'].values
            
            # Create line segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Normalize velocity for colormap
            norm = Normalize(vmin=speed.min(), vmax=speed.max())
            
            # Create LineCollection
            lc = LineCollection(segments, cmap=self.colormap_name, norm=norm,
                              linewidth=self.line_thickness, alpha=0.9)
            lc.set_array(speed[:-1])  # Color by velocity of starting point
            
            ax.add_collection(lc)
            
            # Add colorbar
            cbar = self.figure.colorbar(lc, ax=ax, pad=0.02)
            cbar.set_label('Velocidad (m/s)', fontsize=9)
            
            # Set axis limits manually since add_collection doesn't auto-scale
            margin_x = (x.max() - x.min()) * 0.1 + 0.01
            margin_y = (y.max() - y.min()) * 0.1 + 0.01
            ax.set_xlim(x.min() - margin_x, x.max() + margin_x)
            ax.set_ylim(y.min() - margin_y, y.max() + margin_y)
            
            title += " (color = velocidad)"
        else:
            # Simple line plot
            ax.plot(x, y, color=config.get("color", "#2c3e50"), 
                   linewidth=self.line_thickness, alpha=0.9)
            
            # Add markers for start and end
            if len(x) > 0:
                ax.scatter([x[0]], [y[0]], color='#27ae60', s=100, zorder=5, 
                          marker='o', label='Inicio')
                ax.scatter([x[-1]], [y[-1]], color='#e74c3c', s=100, zorder=5, 
                          marker='s', label='Fin')
                ax.legend(loc='best', fontsize=8)
        
        # Current position marker
        if len(x) > 0:
            # Find closest point to current time
            times = self.metrics_df['time_s'].values
            idx = np.argmin(np.abs(times - self.current_time))
            self.current_position_marker = ax.scatter([x[idx]], [y[idx]], 
                                                      color='red', s=150, zorder=10,
                                                      marker='o', edgecolors='white',
                                                      linewidths=2)
        
        # Labels and title
        ax.set_xlabel('Posición X (m)', fontsize=10)
        ax.set_ylabel('Altura (m)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio without conflicting with xlim/ylim
        ax.set_aspect('equal', adjustable='box')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_time_indicator(self, time_s: float):
        """Update the time indicator (line or position marker)."""
        self.current_time = time_s
        
        current_selection = self.selector.currentText()
        config = GRAPH_OPTIONS.get(current_selection, {})
        
        if config.get("type") == "trajectory":
            # For trajectory, we need to redraw to update position marker
            # This is a bit expensive but ensures correct display
            if self.current_position_marker is not None and not self.metrics_df.empty:
                x = self.metrics_df['x_m'].values
                y = self.metrics_df['height_m'].values
                times = self.metrics_df['time_s'].values
                
                if len(times) > 0:
                    idx = np.argmin(np.abs(times - time_s))
                    self.current_position_marker.set_offsets([[x[idx], y[idx]]])
                    self.canvas.draw_idle()
        else:
            # For time series, just update the vertical line
            if self.time_line:
                self.time_line.set_xdata([time_s, time_s])
                self.canvas.draw_idle()


class VideoPanel(QWidget):
    """Panel for video playback with controls and trimming."""
    
    def __init__(self, video_path: str, fps: float, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.fps = fps
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.playback_speed = 1.0  # 1.0 = normal, 0.25 = slow
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # Trim state
        self.trim_start = 0
        self.trim_end = 0
        
        # Callbacks
        self.on_time_changed_callbacks = []
        self.on_trim_changed_callbacks = []
        
        self.setup_ui()
        self.load_video()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1a1a1a; border-radius: 4px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label)
        
        # Time/Frame info
        self.info_label = QLabel("Frame: 0 / 0  |  Tiempo: 0.00s / 0.00s")
        self.info_label.setFont(QFont("Monaco", 10, QFont.Bold))
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #2c3e50; padding: 5px; background-color: #ecf0f1; border-radius: 4px;")
        layout.addWidget(self.info_label)
        
        # Custom trim slider
        self.trim_slider = TrimSlider()
        self.trim_slider.on_value_changed = self.on_slider_value_changed
        self.trim_slider.on_trim_changed = self.on_trim_range_changed
        layout.addWidget(self.trim_slider)
        
        # Trim info label
        self.trim_info = QLabel("Recorte: Todo el video")
        self.trim_info.setAlignment(Qt.AlignCenter)
        self.trim_info.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        layout.addWidget(self.trim_info)
        
        # Main controls row
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(5)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet(BUTTON_DANGER_STYLE)
        self.stop_btn.clicked.connect(self.stop)
        controls_layout.addWidget(self.stop_btn)
        
        # Play button
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setStyleSheet(BUTTON_STYLE)
        self.play_btn.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_btn)
        
        # Slow motion button
        self.slow_btn = QPushButton("🐢 0.25x")
        self.slow_btn.setStyleSheet(BUTTON_STYLE)
        self.slow_btn.clicked.connect(self.toggle_slow_motion)
        controls_layout.addWidget(self.slow_btn)
        
        layout.addLayout(controls_layout)
        
        # Navigation controls row
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(5)
        
        self.prev_btn = QPushButton("◀◀ -10")
        self.prev_btn.setStyleSheet(BUTTON_STYLE)
        self.prev_btn.clicked.connect(lambda: self.skip_frames(-10))
        nav_layout.addWidget(self.prev_btn)
        
        self.prev_one_btn = QPushButton("◀ -1")
        self.prev_one_btn.setStyleSheet(BUTTON_STYLE)
        self.prev_one_btn.clicked.connect(lambda: self.skip_frames(-1))
        nav_layout.addWidget(self.prev_one_btn)
        
        self.next_one_btn = QPushButton("+1 ▶")
        self.next_one_btn.setStyleSheet(BUTTON_STYLE)
        self.next_one_btn.clicked.connect(lambda: self.skip_frames(1))
        nav_layout.addWidget(self.next_one_btn)
        
        self.next_btn = QPushButton("+10 ▶▶")
        self.next_btn.setStyleSheet(BUTTON_STYLE)
        self.next_btn.clicked.connect(lambda: self.skip_frames(10))
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
        # Trim controls row
        trim_layout = QHBoxLayout()
        trim_layout.setSpacing(5)
        
        self.set_start_btn = QPushButton("📍 Marcar Inicio")
        self.set_start_btn.setStyleSheet(BUTTON_STYLE.replace("#3498db", "#27ae60").replace("#2980b9", "#219a52").replace("#1f618d", "#1e8449"))
        self.set_start_btn.clicked.connect(self.set_trim_start)
        trim_layout.addWidget(self.set_start_btn)
        
        self.set_end_btn = QPushButton("📍 Marcar Fin")
        self.set_end_btn.setStyleSheet(BUTTON_STYLE.replace("#3498db", "#e74c3c").replace("#2980b9", "#c0392b").replace("#1f618d", "#a93226"))
        self.set_end_btn.clicked.connect(self.set_trim_end)
        trim_layout.addWidget(self.set_end_btn)
        
        self.reset_trim_btn = QPushButton("↺ Resetear Recorte")
        self.reset_trim_btn.setStyleSheet(BUTTON_STYLE.replace("#3498db", "#95a5a6").replace("#2980b9", "#7f8c8d").replace("#1f618d", "#707b7c"))
        self.reset_trim_btn.clicked.connect(self.reset_trim)
        trim_layout.addWidget(self.reset_trim_btn)
        
        layout.addLayout(trim_layout)
    
    def load_video(self):
        """Load the video file."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.video_label.setText(f"Error loading video:\n{self.video_path}")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.trim_start = 0
        self.trim_end = self.total_frames - 1
        
        self.trim_slider.setRange(0, self.total_frames - 1)
        self.trim_slider.setTrimRange(0, self.total_frames - 1)
        
        # Display first frame
        self.show_frame(0)
    
    def show_frame(self, frame_idx: int):
        """Display a specific frame."""
        if self.cap is None:
            return
        
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.current_frame = frame_idx
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Scale to fit label while maintaining aspect ratio
            label_size = self.video_label.size()
            h, w = frame_rgb.shape[:2]
            
            # Calculate scale
            scale_w = label_size.width() / w if w > 0 else 1
            scale_h = label_size.height() / h if h > 0 else 1
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if new_w > 0 and new_h > 0:
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                
                # Convert to QImage and display
                bytes_per_line = 3 * new_w
                q_img = QImage(frame_resized.data, new_w, new_h, 
                              bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))
        
        # Update slider
        self.trim_slider.setValue(frame_idx)
        
        # Update info label
        current_time = frame_idx / self.fps
        total_time = self.total_frames / self.fps
        trim_indicator = ""
        if self.trim_slider.is_trimmed:
            trim_indicator = " 🔒"
        
        speed_indicator = ""
        if self.playback_speed != 1.0:
            speed_indicator = f" [{self.playback_speed}x]"
        
        self.info_label.setText(
            f"Frame: {frame_idx:4d} / {self.total_frames}{trim_indicator}  |  "
            f"Tiempo: {current_time:.2f}s / {total_time:.2f}s{speed_indicator}"
        )
        
        # Notify callbacks
        for callback in self.on_time_changed_callbacks:
            callback(current_time)
    
    def toggle_play(self):
        """Toggle play/pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """Start playback."""
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        self.play_btn.setStyleSheet(BUTTON_ACTIVE_STYLE)
        interval = int(1000 / (self.fps * self.playback_speed))
        self.timer.start(interval)
    
    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.play_btn.setStyleSheet(BUTTON_STYLE)
        self.timer.stop()
    
    def stop(self):
        """Stop playback and go to beginning (or trim start)."""
        self.pause()
        start_frame = self.trim_start if self.trim_slider.is_trimmed else 0
        self.show_frame(start_frame)
    
    def toggle_slow_motion(self):
        """Toggle slow motion mode."""
        if self.playback_speed == 1.0:
            self.playback_speed = 0.25
            self.slow_btn.setText("🐢 0.25x ✓")
            self.slow_btn.setStyleSheet(BUTTON_ACTIVE_STYLE)
        else:
            self.playback_speed = 1.0
            self.slow_btn.setText("🐢 0.25x")
            self.slow_btn.setStyleSheet(BUTTON_STYLE)
        
        # Update timer if playing
        if self.is_playing:
            interval = int(1000 / (self.fps * self.playback_speed))
            self.timer.setInterval(interval)
    
    def next_frame(self):
        """Advance to next frame."""
        end_frame = self.trim_end if self.trim_slider.is_trimmed else self.total_frames - 1
        
        if self.current_frame < end_frame:
            self.show_frame(self.current_frame + 1)
        else:
            # Loop to beginning of trim region
            self.pause()
            start_frame = self.trim_start if self.trim_slider.is_trimmed else 0
            self.show_frame(start_frame)
    
    def skip_frames(self, delta: int):
        """Skip forward or backward by delta frames."""
        new_frame = self.current_frame + delta
        self.show_frame(new_frame)
    
    def on_slider_value_changed(self, value: int):
        """Handle slider value change."""
        self.show_frame(value)
    
    def on_trim_range_changed(self, start: int, end: int):
        """Handle trim range change from slider."""
        self.trim_start = start
        self.trim_end = end
        self._update_trim_info()
        
        # Notify callbacks
        for callback in self.on_trim_changed_callbacks:
            callback(start, end)
    
    def set_trim_start(self):
        """Set trim start to current frame."""
        self.trim_start = self.current_frame
        self.trim_slider.setTrimRange(self.trim_start, self.trim_end)
        self._update_trim_info()
        for callback in self.on_trim_changed_callbacks:
            callback(self.trim_start, self.trim_end)
    
    def set_trim_end(self):
        """Set trim end to current frame."""
        self.trim_end = self.current_frame
        self.trim_slider.setTrimRange(self.trim_start, self.trim_end)
        self._update_trim_info()
        for callback in self.on_trim_changed_callbacks:
            callback(self.trim_start, self.trim_end)
    
    def reset_trim(self):
        """Reset trim to full video."""
        self.trim_start = 0
        self.trim_end = self.total_frames - 1
        self.trim_slider.resetTrim()
        self._update_trim_info()
        for callback in self.on_trim_changed_callbacks:
            callback(self.trim_start, self.trim_end)
    
    def _update_trim_info(self):
        """Update the trim info label."""
        if self.trim_slider.is_trimmed:
            start_time = self.trim_start / self.fps
            end_time = self.trim_end / self.fps
            duration = end_time - start_time
            self.trim_info.setText(
                f"Recorte: Frame {self.trim_start} - {self.trim_end} "
                f"({start_time:.2f}s - {end_time:.2f}s, duración: {duration:.2f}s)"
            )
            self.trim_info.setStyleSheet("color: #27ae60; font-size: 10px; font-weight: bold;")
        else:
            self.trim_info.setText("Recorte: Todo el video")
            self.trim_info.setStyleSheet("color: #7f8c8d; font-size: 10px;")
    
    def add_time_callback(self, callback):
        """Add a callback to be called when time changes."""
        self.on_time_changed_callbacks.append(callback)
    
    def add_trim_callback(self, callback):
        """Add a callback to be called when trim range changes."""
        self.on_trim_changed_callbacks.append(callback)
    
    def cleanup(self):
        """Release video resources."""
        if self.cap:
            self.cap.release()


class InteractiveAnalysisViewer(QMainWindow):
    """
    Main window for interactive analysis visualization.
    
    3-panel layout:
    - Left: Graph selector + graph
    - Center: Video with controls
    - Right: Graph selector + graph
    """
    
    def __init__(self, video_path: str, metrics_df: pd.DataFrame, 
                 fps: float = 30.0, title: str = "Análisis de Levantamiento"):
        super().__init__()
        
        self.video_path = video_path
        self.metrics_df = metrics_df
        self.fps = fps
        
        self.setWindowTitle(title)
        self.setStyleSheet("background-color: white;")
        
        self.setup_ui()
        self.resize_to_video()
    
    def setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Left panel (graph)
        self.left_graph = GraphPanel(self.metrics_df)
        self.left_graph.setMinimumWidth(380)
        main_layout.addWidget(self.left_graph, stretch=1)
        
        # Separator
        left_sep = QFrame()
        left_sep.setFrameShape(QFrame.VLine)
        left_sep.setStyleSheet("color: #ddd;")
        main_layout.addWidget(left_sep)
        
        # Center panel (video)
        center_layout = QVBoxLayout()
        center_layout.setSpacing(10)
        
        self.video_panel = VideoPanel(self.video_path, self.fps)
        self.video_panel.add_time_callback(self.on_video_time_changed)
        self.video_panel.add_trim_callback(self.on_trim_changed)
        center_layout.addWidget(self.video_panel, stretch=1)
        
        # Close button
        close_btn = QPushButton("✕ Cerrar Visualización")
        close_btn.setFont(QFont("Arial", 11))
        close_btn.setStyleSheet(BUTTON_DANGER_STYLE)
        close_btn.clicked.connect(self.close)
        center_layout.addWidget(close_btn)
        
        center_widget = QWidget()
        center_widget.setLayout(center_layout)
        main_layout.addWidget(center_widget, stretch=2)
        
        # Separator
        right_sep = QFrame()
        right_sep.setFrameShape(QFrame.VLine)
        right_sep.setStyleSheet("color: #ddd;")
        main_layout.addWidget(right_sep)
        
        # Right panel (graph) - default to trajectory
        self.right_graph = GraphPanel(self.metrics_df)
        self.right_graph.setMinimumWidth(380)
        # Set trajectory as default for right panel
        self.right_graph.selector.setCurrentText("📍 Trayectoria X-Y")
        main_layout.addWidget(self.right_graph, stretch=1)
    
    def resize_to_video(self):
        """Resize window based on video dimensions."""
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Calculate window size
            aspect = width / height if height > 0 else 1
            
            if aspect > 1:  # Horizontal video
                video_width = 640
                video_height = int(640 / aspect)
            else:  # Vertical video
                video_height = 700
                video_width = int(700 * aspect)
            
            # Total width = left_graph + video + right_graph + margins
            total_width = 420 + video_width + 420 + 120
            total_height = max(750, video_height + 280)
            
            self.resize(total_width, total_height)
    
    def on_video_time_changed(self, time_s: float):
        """Update graph time indicators when video time changes."""
        self.left_graph.update_time_indicator(time_s)
        self.right_graph.update_time_indicator(time_s)
    
    def on_trim_changed(self, start_frame: int, end_frame: int):
        """Update graphs when trim range changes."""
        if start_frame == 0 and end_frame == self.video_panel.total_frames - 1:
            # Reset to full data
            self.left_graph.reset_trim()
            self.right_graph.reset_trim()
        else:
            # Apply trim
            self.left_graph.set_trim_range(start_frame, end_frame)
            self.right_graph.set_trim_range(start_frame, end_frame)
    
    def closeEvent(self, event):
        """Clean up on close."""
        self.video_panel.cleanup()
        event.accept()


def launch_interactive_viewer(video_path: str, metrics_df: pd.DataFrame, 
                             fps: float = 30.0, title: str = "Análisis de Levantamiento"):
    """
    Launch the interactive viewer as a standalone application.
    
    Args:
        video_path: Path to the video file (with tracking overlay)
        metrics_df: DataFrame with metrics from MetricsCalculator
        fps: Video frame rate
        title: Window title
    """
    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set palette for better text visibility
    palette = app.palette()
    palette.setColor(QPalette.WindowText, QColor("#333333"))
    palette.setColor(QPalette.Text, QColor("#333333"))
    palette.setColor(QPalette.ButtonText, QColor("#333333"))
    app.setPalette(palette)
    
    # Create and show viewer
    viewer = InteractiveAnalysisViewer(video_path, metrics_df, fps, title)
    viewer.show()
    
    # Run event loop
    return app.exec_()


# Pipeline integration
class InteractiveViewerLauncher:
    """
    Pipeline step that launches the interactive viewer.
    
    This is a terminal step that opens the GUI when the pipeline completes.
    """
    
    def __init__(self):
        self.config = {}
    
    def run(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Launch the interactive viewer.
        
        Args:
            input_data: Dict containing:
                - metrics_df: pandas DataFrame from MetricsCalculator
                - video_path: Path to the tracking video
            config: Configuration with optional title, fps, etc.
        """
        self.config = config
        
        metrics_df = input_data.get("metrics_df")
        video_path = input_data.get("video_path")
        
        if metrics_df is None or video_path is None:
            print("[InteractiveViewer] Error: Missing metrics_df or video_path")
            return
        
        fps = config.get("fps", 30.0)
        title = config.get("title", "Análisis de Levantamiento")
        
        print(f"[InteractiveViewer] Launching interactive viewer...")
        print(f"[InteractiveViewer] Video: {video_path}")
        print(f"[InteractiveViewer] Metrics: {len(metrics_df)} rows")
        
        launch_interactive_viewer(video_path, metrics_df, fps, title)
    
    def save_result(self, data, output_path):
        """No output to save."""
        pass
    
    def load_result(self, input_path):
        """No output to load."""
        return None


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    import numpy as np
    
    # Create sample metrics
    n_frames = 100
    time = np.linspace(0, 3.5, n_frames)
    
    sample_df = pd.DataFrame({
        'frame_idx': np.arange(n_frames),
        'time_s': time,
        'x_m': 0.5 + 0.1 * np.sin(time * 2),
        'y_m': 1.0 + 0.5 * np.sin(time * 1.5),
        'height_m': 0.5 * (1 - np.cos(time * 1.8)),
        'vx_m_s': 0.2 * np.cos(time * 2),
        'vy_m_s': 0.75 * np.cos(time * 1.5),
        'speed_m_s': np.abs(0.8 * np.cos(time * 1.5)),
        'ax_m_s2': -0.4 * np.sin(time * 2),
        'ay_m_s2': -1.1 * np.sin(time * 1.5),
        'accel_m_s2': np.abs(1.2 * np.sin(time * 1.5)),
        'kinetic_energy_j': 50 * (0.8 * np.cos(time * 1.5))**2,
        'potential_energy_j': 100 * 9.81 * 0.5 * (1 - np.cos(time * 1.8)),
        'total_energy_j': 50 * (0.8 * np.cos(time * 1.5))**2 + 100 * 9.81 * 0.5 * (1 - np.cos(time * 1.8)),
        'power_w': 500 * np.sin(time * 3),
    })
    
    print("Launching test viewer...")
    print("(Press close button or Ctrl+C to exit)")
    
    # This would need a real video path to work
    # launch_interactive_viewer("test_video.mp4", sample_df, 30.0)
