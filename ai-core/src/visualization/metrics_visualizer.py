"""
MetricsVisualizer: Creates plots from physical metrics DataFrame.

Generates a multi-panel figure with:
- Position (height) vs time
- Velocity vs time
- Acceleration vs time
- Energy (kinetic, potential, total) vs time
- Power vs time

Output: PNG image file with all plots in a single panel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from domain.ports import IPipelineStep


class MetricsVisualizer(IPipelineStep[pd.DataFrame, str]):
    """
    Creates visualization plots from metrics DataFrame.
    
    Config:
        output_filename: str (path for output image)
        
        # Plot configuration
        figsize: [width, height] (default [14, 12])
        dpi: int (default 150)
        style: str (default "seaborn-v0_8-whitegrid")
        
        # Which plots to include
        plots:
            position: bool (default true)
            velocity: bool (default true)
            acceleration: bool (default true)
            energy: bool (default true)
            power: bool (default true)
        
        # Colors
        colors:
            position: "#2ecc71"
            velocity: "#3498db"
            acceleration: "#e74c3c"
            kinetic_energy: "#9b59b6"
            potential_energy: "#f39c12"
            total_energy: "#1abc9c"
            power: "#e91e63"
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
    
    def run(self, input_data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """
        Generate metrics visualization.
        
        Args:
            input_data: DataFrame from MetricsCalculator
            config: Visualization configuration
            
        Returns:
            Path to output image file
        """
        self.config = config
        
        if input_data.empty:
            print("[MetricsVisualizer] Warning: Empty DataFrame, skipping visualization")
            return ""
        
        # Configuration
        output_filename = config.get("output_filename", "metrics_plot.png")
        figsize = config.get("figsize", [14, 12])
        dpi = config.get("dpi", 150)
        
        # Try to set style, fall back if not available
        style = config.get("style", "seaborn-v0_8-whitegrid")
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use("seaborn-whitegrid")
            except:
                pass  # Use default
        
        # Plot configuration
        plots_config = config.get("plots", {})
        show_position = plots_config.get("position", True)
        show_velocity = plots_config.get("velocity", True)
        show_acceleration = plots_config.get("acceleration", True)
        show_energy = plots_config.get("energy", True)
        show_power = plots_config.get("power", True)
        
        # Colors
        colors = config.get("colors", {})
        color_position = colors.get("position", "#2ecc71")
        color_velocity = colors.get("velocity", "#3498db")
        color_acceleration = colors.get("acceleration", "#e74c3c")
        color_ke = colors.get("kinetic_energy", "#9b59b6")
        color_pe = colors.get("potential_energy", "#f39c12")
        color_te = colors.get("total_energy", "#1abc9c")
        color_power = colors.get("power", "#e91e63")
        
        # Count active plots
        active_plots = sum([show_position, show_velocity, show_acceleration, 
                          show_energy, show_power])
        
        if active_plots == 0:
            print("[MetricsVisualizer] Warning: No plots enabled")
            return ""
        
        print(f"[MetricsVisualizer] Creating {active_plots} plots...")
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(active_plots, 1, hspace=0.3)
        
        time = input_data['time_s'].values
        plot_idx = 0
        
        # --- Position (Height) ---
        if show_position:
            ax = fig.add_subplot(gs[plot_idx])
            ax.plot(time, input_data['height_m'], color=color_position, 
                   linewidth=2, label='Altura')
            ax.set_ylabel('Altura (m)', fontsize=11)
            ax.set_xlabel('Tiempo (s)', fontsize=11)
            ax.set_title('Altura del Disco vs Tiempo', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plot_idx += 1
        
        # --- Velocity ---
        if show_velocity:
            ax = fig.add_subplot(gs[plot_idx])
            ax.plot(time, input_data['speed_m_s'], color=color_velocity, 
                   linewidth=2, label='Velocidad')
            ax.fill_between(time, 0, input_data['speed_m_s'], 
                           color=color_velocity, alpha=0.2)
            ax.set_ylabel('Velocidad (m/s)', fontsize=11)
            ax.set_xlabel('Tiempo (s)', fontsize=11)
            ax.set_title('Velocidad del Disco vs Tiempo', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Mark peak velocity
            peak_idx = input_data['speed_m_s'].idxmax()
            peak_time = input_data.loc[peak_idx, 'time_s']
            peak_speed = input_data.loc[peak_idx, 'speed_m_s']
            ax.annotate(f'Pico: {peak_speed:.2f} m/s', 
                       xy=(peak_time, peak_speed),
                       xytext=(peak_time + 0.1, peak_speed * 1.1),
                       fontsize=9, color=color_velocity,
                       arrowprops=dict(arrowstyle='->', color=color_velocity))
            plot_idx += 1
        
        # --- Acceleration ---
        if show_acceleration:
            ax = fig.add_subplot(gs[plot_idx])
            ax.plot(time, input_data['accel_m_s2'], color=color_acceleration, 
                   linewidth=2, label='Aceleración')
            ax.fill_between(time, 0, input_data['accel_m_s2'], 
                           color=color_acceleration, alpha=0.2)
            ax.set_ylabel('Aceleración (m/s²)', fontsize=11)
            ax.set_xlabel('Tiempo (s)', fontsize=11)
            ax.set_title('Aceleración del Disco vs Tiempo', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=9.81, color='gray', linestyle='--', alpha=0.5, label='g')
            plot_idx += 1
        
        # --- Energy ---
        if show_energy:
            ax = fig.add_subplot(gs[plot_idx])
            ax.plot(time, input_data['kinetic_energy_j'], color=color_ke, 
                   linewidth=2, label='Energía Cinética', linestyle='-')
            ax.plot(time, input_data['potential_energy_j'], color=color_pe, 
                   linewidth=2, label='Energía Potencial', linestyle='--')
            ax.plot(time, input_data['total_energy_j'], color=color_te, 
                   linewidth=2.5, label='Energía Total', linestyle='-')
            ax.set_ylabel('Energía (J)', fontsize=11)
            ax.set_xlabel('Tiempo (s)', fontsize=11)
            ax.set_title('Energía Mecánica vs Tiempo', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # --- Power ---
        if show_power:
            ax = fig.add_subplot(gs[plot_idx])
            
            # Separate positive and negative power
            power = input_data['power_w'].values
            pos_power = np.where(power > 0, power, 0)
            neg_power = np.where(power < 0, power, 0)
            
            ax.fill_between(time, 0, pos_power, color=color_power, alpha=0.4, 
                           label='Potencia (+)')
            ax.fill_between(time, 0, neg_power, color='#607d8b', alpha=0.4, 
                           label='Potencia (-)')
            ax.plot(time, power, color=color_power, linewidth=1.5)
            
            ax.set_ylabel('Potencia (W)', fontsize=11)
            ax.set_xlabel('Tiempo (s)', fontsize=11)
            ax.set_title('Potencia Instantánea vs Tiempo', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            
            # Mark peak power
            peak_power_idx = input_data['power_w'].idxmax()
            peak_time = input_data.loc[peak_power_idx, 'time_s']
            peak_power = input_data.loc[peak_power_idx, 'power_w']
            ax.annotate(f'Pico: {peak_power:.0f} W', 
                       xy=(peak_time, peak_power),
                       xytext=(peak_time + 0.1, peak_power * 1.1),
                       fontsize=9, color=color_power,
                       arrowprops=dict(arrowstyle='->', color=color_power))
        
        # Add title with summary stats
        if 'speed_m_s' in input_data.columns and 'power_w' in input_data.columns:
            peak_speed = input_data['speed_m_s'].max()
            peak_power = input_data['power_w'].max()
            max_height = input_data['height_m'].max()
            
            fig.suptitle(
                f'Análisis de Levantamiento | Vel. Pico: {peak_speed:.2f} m/s | '
                f'Potencia Pico: {peak_power:.0f} W | Altura Máx: {max_height:.2f} m',
                fontsize=13, fontweight='bold', y=1.02
            )
        
        # Resolve output path
        output_path = Path(output_filename)
        if not output_path.is_absolute():
            workspace = Path(config.get("_workspace_root", "."))
            output_path = workspace / output_filename
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"[MetricsVisualizer] Saved plot: {output_path}")
        
        return str(output_path)
    
    def save_result(self, data: str, output_path: Path) -> None:
        """Save output path reference."""
        result = {"plot_path": data}
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def load_result(self, input_path: Path) -> str:
        """Load output path from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        return data.get("plot_path", "")
