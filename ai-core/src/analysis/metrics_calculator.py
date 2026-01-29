"""
MetricsCalculator: Computes physical metrics from tracked disc trajectory.

This module takes the trajectory (x, y positions per frame) and computes:
- Position in meters (using disc diameter as scale reference)
- Velocity (m/s)
- Acceleration (m/s²)
- Kinetic Energy (J)
- Potential Energy (J)
- Power (W)

Physical parameters (athlete height, disc size, disc weight, bar weight) are
provided via external configuration (YAML).

Output: pandas DataFrame with one row per frame, columns for each metric.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from domain.ports import IPipelineStep
from domain.entities import TrackedObject


@dataclass
class PhysicalParams:
    """Physical parameters for metrics calculation."""
    athlete_height_m: float = 1.75      # Athlete height in meters
    disc_diameter_m: float = 0.45       # Standard Olympic disc = 45cm
    disc_weight_kg: float = 20.0        # Weight of ONE disc
    bar_weight_kg: float = 20.0         # Olympic bar = 20kg
    num_discs: int = 2                  # Discs on each side (total = 2 * num_discs)
    gravity: float = 9.81               # m/s²
    
    @property
    def total_weight_kg(self) -> float:
        """Total weight being lifted."""
        return self.bar_weight_kg + (2 * self.num_discs * self.disc_weight_kg)
    
    @property
    def disc_radius_m(self) -> float:
        return self.disc_diameter_m / 2.0


class MetricsCalculator(IPipelineStep[Dict[int, List[TrackedObject]], pd.DataFrame]):
    """
    Calculates physical metrics from tracked object trajectory.
    
    Config:
        # Physical parameters (can be in separate YAML)
        physical_params:
            athlete_height_m: 1.75
            disc_diameter_m: 0.45
            disc_weight_kg: 20.0
            bar_weight_kg: 20.0
            num_discs: 2
        
        # Or load from file
        physical_params_file: "params.yaml"
        
        # Which class to analyze
        target_class: "discos"  # or "disco", "frisbee"
        
        # Reference for scale (from disc selection)
        reference_radius_px: float (if not provided, estimated from detections)
        selection_file: str (optional, load from disc_selection.json)
        
        # Video info (usually auto-detected)
        fps: float (frames per second)
        total_frames: int
        
        # Smoothing before differentiation
        smooth_window: int (default 5, for velocity/acceleration)
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.params: PhysicalParams = PhysicalParams()
        self.scale_m_per_px: float = 1.0
        self.fps: float = 30.0
        self.dt: float = 1.0 / 30.0
    
    def run(self, input_data: Dict[int, List[TrackedObject]], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate metrics from tracked objects.
        
        Args:
            input_data: Dict[frame_idx, List[TrackedObject]] from ModelTracker
            config: Configuration with physical parameters
            
        Returns:
            pd.DataFrame with metrics per frame
        """
        self.config = config
        
        # Load physical parameters
        self._load_physical_params(config)
        
        # Get video timing info - prefer injected metadata over config
        # Priority: _video_fps (from pipeline) > fps (from config) > default 30.0
        if config.get("_video_fps") and config["_video_fps"] > 0:
            self.fps = config["_video_fps"]
            print(f"[MetricsCalculator] Using video metadata FPS: {self.fps}")
        else:
            self.fps = config.get("fps", 30.0)
            print(f"[MetricsCalculator] Using config FPS: {self.fps}")
        
        self.dt = 1.0 / self.fps
        
        # Total frames from metadata or config
        total_frames = config.get("_video_total_frames") or config.get("total_frames", len(input_data))
        
        # Store video metadata for potential use
        self.video_width = config.get("_video_width", 0)
        self.video_height = config.get("_video_height", 0)
        self.video_duration = config.get("_video_duration", 0.0)
        
        # Get target class
        target_class = config.get("target_class", "discos")
        target_classes = [target_class]
        
        # Handle label mapping (disco -> discos, frisbee, etc.)
        if target_class == "disco":
            target_classes = ["disco", "discos", "frisbee"]
        
        # Get reference radius for scale
        reference_radius_px = self._get_reference_radius(config)
        
        if reference_radius_px and reference_radius_px > 0:
            self.scale_m_per_px = self.params.disc_radius_m / reference_radius_px
            print(f"[MetricsCalculator] Scale: {self.scale_m_per_px:.6f} m/px")
            print(f"[MetricsCalculator] Reference radius: {reference_radius_px:.1f} px = {self.params.disc_radius_m:.3f} m")
        else:
            print(f"[MetricsCalculator] Warning: No reference radius, using 1 px = 0.001 m")
            self.scale_m_per_px = 0.001
        
        print(f"[MetricsCalculator] Physical params:")
        print(f"  - Total weight: {self.params.total_weight_kg:.1f} kg")
        print(f"  - Disc diameter: {self.params.disc_diameter_m:.3f} m")
        print(f"  - FPS: {self.fps}, dt: {self.dt:.4f} s")
        
        # Extract trajectory for target class
        trajectory = self._extract_trajectory(input_data, target_classes)
        
        if len(trajectory) < 2:
            print(f"[MetricsCalculator] Warning: Insufficient trajectory points ({len(trajectory)})")
            return pd.DataFrame()
        
        print(f"[MetricsCalculator] Trajectory: {len(trajectory)} points")
        
        # Calculate metrics
        smooth_window = config.get("smooth_window", 5)
        df = self._calculate_metrics(trajectory, smooth_window)
        
        print(f"[MetricsCalculator] Computed {len(df)} rows of metrics")
        
        return df
    
    def _load_physical_params(self, config: Dict[str, Any]) -> None:
        """Load physical parameters from config or file."""
        # Try loading from file first
        params_file = config.get("physical_params_file")
        if params_file:
            params_path = Path(params_file)
            if not params_path.is_absolute():
                workspace = Path(config.get("_workspace_root", "."))
                params_path = workspace / params_file
            
            if params_path.exists():
                import yaml
                with open(params_path, 'r') as f:
                    file_params = yaml.safe_load(f)
                if file_params:
                    self.params = PhysicalParams(**file_params)
                    print(f"[MetricsCalculator] Loaded params from: {params_path}")
                    return
        
        # Otherwise load from inline config
        inline_params = config.get("physical_params", {})
        if inline_params:
            self.params = PhysicalParams(
                athlete_height_m=inline_params.get("athlete_height_m", 1.75),
                disc_diameter_m=inline_params.get("disc_diameter_m", 0.45),
                disc_weight_kg=inline_params.get("disc_weight_kg", 20.0),
                bar_weight_kg=inline_params.get("bar_weight_kg", 20.0),
                num_discs=inline_params.get("num_discs", 2),
            )
    
    def _get_reference_radius(self, config: Dict[str, Any]) -> Optional[float]:
        """Get reference radius in pixels for scale calculation."""
        # Direct config
        if config.get("reference_radius_px"):
            return float(config["reference_radius_px"])
        
        # From selection file - check both user config and injected path
        selection_file = config.get("selection_file") or config.get("_selection_file")
        if selection_file:
            selection_path = Path(selection_file)
            if not selection_path.is_absolute():
                workspace = Path(config.get("_project_root", "."))
                selection_path = workspace / "ai-core" / selection_file
            
            if selection_path.exists():
                with open(selection_path, 'r') as f:
                    selection = json.load(f)
                if selection.get("radius"):
                    print(f"[MetricsCalculator] Loaded reference radius from: {selection_path}")
                    return float(selection["radius"])
            else:
                print(f"[MetricsCalculator] Warning: Selection file not found: {selection_path}")
        
        return None
    
    def _extract_trajectory(self, data: Dict[int, List[TrackedObject]], 
                           target_classes: List[str]) -> List[Tuple[int, float, float]]:
        """
        Extract trajectory for target class.
        Returns list of (frame_idx, x_px, y_px).
        """
        trajectory = []
        
        for frame_idx in sorted(data.keys()):
            tracked_objs = data[frame_idx]
            
            for obj in tracked_objs:
                if obj.detection.class_name in target_classes:
                    # Get center position
                    bbox = obj.detection.bbox
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    
                    # Use smoothed position if available
                    if obj.smoothed_position:
                        cx, cy = obj.smoothed_position
                    
                    trajectory.append((frame_idx, cx, cy))
                    break  # Only one disc per frame (single-object tracking)
        
        return trajectory
    
    def _calculate_metrics(self, trajectory: List[Tuple[int, float, float]], 
                          smooth_window: int) -> pd.DataFrame:
        """
        Calculate all physical metrics from trajectory.
        
        Returns DataFrame with columns:
        - frame_idx: Frame number
        - time_s: Time in seconds
        - x_px, y_px: Position in pixels
        - x_m, y_m: Position in meters
        - vx_m_s, vy_m_s: Velocity components
        - speed_m_s: Speed magnitude
        - ax_m_s2, ay_m_s2: Acceleration components
        - accel_m_s2: Acceleration magnitude
        - height_m: Height (from initial position)
        - kinetic_energy_j: Kinetic energy
        - potential_energy_j: Potential energy
        - total_energy_j: Total mechanical energy
        - power_w: Instantaneous power
        """
        # Convert to arrays
        frames = np.array([t[0] for t in trajectory])
        x_px = np.array([t[1] for t in trajectory])
        y_px = np.array([t[2] for t in trajectory])
        
        # Convert to meters
        x_m = x_px * self.scale_m_per_px
        y_m = y_px * self.scale_m_per_px
        
        # Time array
        time_s = frames * self.dt
        
        # Height relative to initial position (Y axis is inverted in image)
        # Lower Y in image = higher in real world
        y_initial = y_m[0]
        height_m = y_initial - y_m  # Positive when disc goes up
        
        # Smooth positions before differentiation
        if smooth_window > 1 and len(x_m) >= smooth_window:
            x_m_smooth = self._smooth(x_m, smooth_window)
            y_m_smooth = self._smooth(y_m, smooth_window)
        else:
            x_m_smooth = x_m
            y_m_smooth = y_m
        
        # Velocity (central difference where possible)
        vx_m_s = np.gradient(x_m_smooth, self.dt)
        vy_m_s = np.gradient(y_m_smooth, self.dt)
        speed_m_s = np.sqrt(vx_m_s**2 + vy_m_s**2)
        
        # Smooth velocity before computing acceleration
        if smooth_window > 1 and len(vx_m_s) >= smooth_window:
            vx_smooth = self._smooth(vx_m_s, smooth_window)
            vy_smooth = self._smooth(vy_m_s, smooth_window)
        else:
            vx_smooth = vx_m_s
            vy_smooth = vy_m_s
        
        # Acceleration
        ax_m_s2 = np.gradient(vx_smooth, self.dt)
        ay_m_s2 = np.gradient(vy_smooth, self.dt)
        accel_m_s2 = np.sqrt(ax_m_s2**2 + ay_m_s2**2)
        
        # Mass
        m = self.params.total_weight_kg
        g = self.params.gravity
        
        # Kinetic energy: KE = 0.5 * m * v²
        kinetic_energy_j = 0.5 * m * speed_m_s**2
        
        # Potential energy: PE = m * g * h
        potential_energy_j = m * g * height_m
        
        # Total mechanical energy
        total_energy_j = kinetic_energy_j + potential_energy_j
        
        # Power: P = F · v = m * a * v (simplified)
        # More accurately: P = d(KE)/dt + d(PE)/dt
        # Using P = m * (a · v) where a and v are vectors
        power_w = m * (ax_m_s2 * vx_m_s + ay_m_s2 * vy_m_s)
        
        # Build DataFrame
        df = pd.DataFrame({
            'frame_idx': frames,
            'time_s': time_s,
            'x_px': x_px,
            'y_px': y_px,
            'x_m': x_m,
            'y_m': y_m,
            'height_m': height_m,
            'vx_m_s': vx_m_s,
            'vy_m_s': vy_m_s,
            'speed_m_s': speed_m_s,
            'ax_m_s2': ax_m_s2,
            'ay_m_s2': ay_m_s2,
            'accel_m_s2': accel_m_s2,
            'kinetic_energy_j': kinetic_energy_j,
            'potential_energy_j': potential_energy_j,
            'total_energy_j': total_energy_j,
            'power_w': power_w,
        })
        
        return df
    
    def _smooth(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(values) < window:
            return values
        
        kernel = np.ones(window) / window
        # Use 'same' mode and handle edges
        smoothed = np.convolve(values, kernel, mode='same')
        
        # Fix edges by using original values
        half = window // 2
        smoothed[:half] = values[:half]
        smoothed[-half:] = values[-half:]
        
        return smoothed
    
    def save_result(self, data: pd.DataFrame, output_path: Path) -> None:
        """Save metrics DataFrame to CSV and JSON."""
        # Save as CSV (human readable)
        csv_path = output_path.with_suffix('.csv')
        data.to_csv(csv_path, index=False)
        print(f"[MetricsCalculator] Saved CSV: {csv_path}")
        
        # Save as JSON (for API/app consumption)
        json_path = output_path  # Keep original .json extension
        data.to_json(json_path, orient='records', indent=2)
        
        # Also save summary statistics
        summary = self._compute_summary(data)
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[MetricsCalculator] Saved summary: {summary_path}")
    
    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics from metrics."""
        if df.empty:
            return {"error": "No data"}
        
        return {
            "trajectory_points": len(df),
            "duration_s": float(df['time_s'].max() - df['time_s'].min()),
            "position": {
                "total_distance_m": float(np.sum(np.sqrt(
                    np.diff(df['x_m'])**2 + np.diff(df['y_m'])**2
                ))),
                "max_height_m": float(df['height_m'].max()),
                "min_height_m": float(df['height_m'].min()),
            },
            "velocity": {
                "peak_speed_m_s": float(df['speed_m_s'].max()),
                "avg_speed_m_s": float(df['speed_m_s'].mean()),
                "peak_vertical_speed_m_s": float(np.abs(df['vy_m_s']).max()),
            },
            "acceleration": {
                "peak_accel_m_s2": float(df['accel_m_s2'].max()),
                "avg_accel_m_s2": float(df['accel_m_s2'].mean()),
            },
            "energy": {
                "peak_kinetic_j": float(df['kinetic_energy_j'].max()),
                "peak_potential_j": float(df['potential_energy_j'].max()),
                "peak_total_j": float(df['total_energy_j'].max()),
            },
            "power": {
                "peak_power_w": float(df['power_w'].max()),
                "avg_power_w": float(df['power_w'].mean()),
                "min_power_w": float(df['power_w'].min()),  # Negative during descent
            },
        }
    
    def load_result(self, input_path: Path) -> pd.DataFrame:
        """Load metrics from file."""
        if input_path.suffix == '.csv':
            return pd.read_csv(input_path)
        else:
            return pd.read_json(input_path)
