"""
TrackRefiner: Post-tracking heuristics for refining track trajectories.

This module applies refinements AFTER tracking is complete.
It operates on full or partial trajectories and can use temporal information.

Refinements:
- Trajectory smoothing: Reduce noise in position estimates
- Outlier removal: Remove sudden jumps that are likely errors
- Direction constraints: Validate trajectory follows expected motion patterns
- Interpolation: Fill gaps in trajectories (future)

Note: This module receives TrackedObjects and outputs refined TrackedObjects.
The refinement preserves track IDs and other metadata.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from domain.ports import IPipelineStep
from domain.entities import TrackedObject, Detection


@dataclass
class TrajectoryPoint:
    """A single point in a trajectory."""
    frame_idx: int
    x: float
    y: float
    score: float = 0.0


@dataclass 
class RefinedTrajectory:
    """A refined trajectory for a track."""
    track_id: int
    class_name: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    smoothed_points: List[TrajectoryPoint] = field(default_factory=list)
    
    # Refinement stats
    outliers_removed: int = 0
    gaps_interpolated: int = 0


def moving_average_smooth(values: List[float], window: int = 5) -> List[float]:
    """
    Apply moving average smoothing to a list of values.
    Uses symmetric window where possible, asymmetric at edges.
    """
    if len(values) <= 1:
        return values.copy()
    
    smoothed = []
    half_window = window // 2
    
    for i in range(len(values)):
        # Calculate window bounds
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        
        # Compute average
        window_values = values[start:end]
        smoothed.append(sum(window_values) / len(window_values))
    
    return smoothed


def exponential_smooth(values: List[float], alpha: float = 0.3) -> List[float]:
    """
    Apply exponential smoothing (EMA) to a list of values.
    alpha: smoothing factor (0-1). Higher = more weight on recent values.
    """
    if len(values) <= 1:
        return values.copy()
    
    smoothed = [values[0]]
    for i in range(1, len(values)):
        new_val = alpha * values[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(new_val)
    
    return smoothed


def savgol_smooth(values: List[float], window: int = 7, order: int = 2) -> List[float]:
    """
    Apply Savitzky-Golay filter for smoothing.
    This preserves peaks and edges better than moving average.
    Requires scipy.
    """
    if len(values) < window:
        return values.copy()
    
    try:
        from scipy.signal import savgol_filter
        # Window must be odd
        if window % 2 == 0:
            window += 1
        return list(savgol_filter(values, window, order))
    except ImportError:
        # Fallback to moving average
        return moving_average_smooth(values, window)


def detect_outliers(values: List[float], threshold_std: float = 3.0) -> List[int]:
    """
    Detect outlier indices based on velocity (rate of change).
    Returns indices of points that are likely outliers.
    """
    if len(values) < 3:
        return []
    
    # Compute velocities (first derivative)
    velocities = [values[i+1] - values[i] for i in range(len(values) - 1)]
    
    if not velocities:
        return []
    
    mean_vel = np.mean(velocities)
    std_vel = np.std(velocities)
    
    if std_vel < 1e-6:
        return []
    
    outliers = []
    for i, vel in enumerate(velocities):
        if abs(vel - mean_vel) > threshold_std * std_vel:
            # Mark the point after the jump as outlier
            outliers.append(i + 1)
    
    return outliers


class TrackRefiner(IPipelineStep[Dict[int, List[TrackedObject]], Dict[int, List[TrackedObject]]]):
    """
    Post-tracking trajectory refinement.
    
    Applies smoothing and validation to tracked trajectories.
    
    Config:
        enabled: bool (default True)
        
        smoothing:
            enabled: bool (default True)
            method: str ("moving_average", "exponential", "savgol")
            window: int (default 5, for moving_average and savgol)
            alpha: float (default 0.3, for exponential)
        
        outlier_removal:
            enabled: bool (default False)
            threshold_std: float (default 3.0)
        
        direction_constraints:  # Future feature
            enabled: bool (default False)
            primary_direction: str ("vertical", "horizontal")
            max_lateral_deviation_px: float
        
        classes_to_refine: List[str] or None (refine all if None)
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.trajectories: Dict[str, Dict[int, RefinedTrajectory]] = {}  # class -> track_id -> trajectory
    
    def run(self, input_data: Dict[int, List[TrackedObject]], config: Dict[str, Any]) -> Dict[int, List[TrackedObject]]:
        """
        Refine tracked objects.
        
        Args:
            input_data: Dict[frame_idx, List[TrackedObject]]
            config: Refinement configuration
            
        Returns:
            Dict[frame_idx, List[TrackedObject]] - Refined tracked objects
        """
        self.config = config
        self.trajectories = {}
        
        if not config.get("enabled", True):
            print("[TrackRefiner] Disabled, passing through")
            return input_data
        
        classes_to_refine = config.get("classes_to_refine")
        smoothing_cfg = config.get("smoothing", {})
        outlier_cfg = config.get("outlier_removal", {})
        
        smoothing_enabled = smoothing_cfg.get("enabled", True)
        smoothing_method = smoothing_cfg.get("method", "moving_average")
        smoothing_window = smoothing_cfg.get("window", 5)
        smoothing_alpha = smoothing_cfg.get("alpha", 0.3)
        
        outlier_enabled = outlier_cfg.get("enabled", False)
        outlier_threshold = outlier_cfg.get("threshold_std", 3.0)
        
        print(f"[TrackRefiner] Refining tracks...")
        if smoothing_enabled:
            print(f"[TrackRefiner] Smoothing: {smoothing_method} (window={smoothing_window})")
        if outlier_enabled:
            print(f"[TrackRefiner] Outlier removal: threshold={outlier_threshold} std")
        
        # Step 1: Collect all trajectories from frame data
        self._collect_trajectories(input_data, classes_to_refine)
        
        # Step 2: Apply refinements to each trajectory
        for class_name, class_trajectories in self.trajectories.items():
            for track_id, trajectory in class_trajectories.items():
                if len(trajectory.points) < 2:
                    # Can't smooth single point
                    trajectory.smoothed_points = trajectory.points.copy()
                    continue
                
                # Extract x and y sequences
                xs = [p.x for p in trajectory.points]
                ys = [p.y for p in trajectory.points]
                
                # Outlier removal (optional)
                if outlier_enabled:
                    x_outliers = set(detect_outliers(xs, outlier_threshold))
                    y_outliers = set(detect_outliers(ys, outlier_threshold))
                    outlier_indices = x_outliers | y_outliers
                    
                    if outlier_indices:
                        # Remove outliers (simple approach: replace with interpolated values)
                        for idx in sorted(outlier_indices, reverse=True):
                            if 0 < idx < len(xs) - 1:
                                # Interpolate from neighbors
                                xs[idx] = (xs[idx-1] + xs[idx+1]) / 2
                                ys[idx] = (ys[idx-1] + ys[idx+1]) / 2
                                trajectory.outliers_removed += 1
                
                # Smoothing
                if smoothing_enabled:
                    if smoothing_method == "moving_average":
                        xs_smooth = moving_average_smooth(xs, smoothing_window)
                        ys_smooth = moving_average_smooth(ys, smoothing_window)
                    elif smoothing_method == "exponential":
                        xs_smooth = exponential_smooth(xs, smoothing_alpha)
                        ys_smooth = exponential_smooth(ys, smoothing_alpha)
                    elif smoothing_method == "savgol":
                        xs_smooth = savgol_smooth(xs, smoothing_window)
                        ys_smooth = savgol_smooth(ys, smoothing_window)
                    else:
                        xs_smooth = xs
                        ys_smooth = ys
                else:
                    xs_smooth = xs
                    ys_smooth = ys
                
                # Build smoothed trajectory
                trajectory.smoothed_points = [
                    TrajectoryPoint(
                        frame_idx=trajectory.points[i].frame_idx,
                        x=xs_smooth[i],
                        y=ys_smooth[i],
                        score=trajectory.points[i].score
                    )
                    for i in range(len(trajectory.points))
                ]
        
        # Step 3: Apply smoothed positions back to output
        results = self._apply_refinements(input_data, classes_to_refine)
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _collect_trajectories(self, data: Dict[int, List[TrackedObject]], 
                               classes_filter: Optional[List[str]]) -> None:
        """Collect trajectories from frame-indexed data."""
        self.trajectories = defaultdict(dict)
        
        for frame_idx in sorted(data.keys()):
            for tracked_obj in data[frame_idx]:
                class_name = tracked_obj.detection.class_name
                track_id = tracked_obj.track_id
                
                # Filter by class if specified
                if classes_filter and class_name not in classes_filter:
                    continue
                
                # Create trajectory if needed
                if track_id not in self.trajectories[class_name]:
                    self.trajectories[class_name][track_id] = RefinedTrajectory(
                        track_id=track_id,
                        class_name=class_name
                    )
                
                # Add point
                bbox = tracked_obj.detection.bbox
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                
                self.trajectories[class_name][track_id].points.append(
                    TrajectoryPoint(
                        frame_idx=frame_idx,
                        x=cx,
                        y=cy,
                        score=tracked_obj.detection.confidence
                    )
                )
    
    def _apply_refinements(self, data: Dict[int, List[TrackedObject]], 
                           classes_filter: Optional[List[str]]) -> Dict[int, List[TrackedObject]]:
        """Apply smoothed positions back to tracked objects."""
        # Build lookup: (class, track_id, frame_idx) -> smoothed position
        smoothed_lookup = {}
        for class_name, class_trajectories in self.trajectories.items():
            for track_id, trajectory in class_trajectories.items():
                for point in trajectory.smoothed_points:
                    key = (class_name, track_id, point.frame_idx)
                    smoothed_lookup[key] = (point.x, point.y)
        
        # Create output with smoothed positions
        results = {}
        for frame_idx, tracked_objs in data.items():
            frame_results = []
            for tracked_obj in tracked_objs:
                class_name = tracked_obj.detection.class_name
                track_id = tracked_obj.track_id
                
                key = (class_name, track_id, frame_idx)
                
                if key in smoothed_lookup:
                    # Apply smoothed position to history
                    smoothed_x, smoothed_y = smoothed_lookup[key]
                    
                    # Create new TrackedObject with updated history
                    # Keep original detection but add smoothed position to history
                    new_history = tracked_obj.history.copy() if tracked_obj.history else []
                    if new_history:
                        # Replace last position with smoothed
                        new_history[-1] = (smoothed_x, smoothed_y)
                    else:
                        new_history = [(smoothed_x, smoothed_y)]
                    
                    refined_obj = TrackedObject(
                        track_id=tracked_obj.track_id,
                        detection=tracked_obj.detection,
                        history=new_history,
                        velocity=tracked_obj.velocity,
                        # Add smoothed position as extra info
                        smoothed_position=(smoothed_x, smoothed_y)
                    )
                    frame_results.append(refined_obj)
                else:
                    # No refinement for this class, pass through
                    frame_results.append(tracked_obj)
            
            results[frame_idx] = frame_results
        
        return results
    
    def _print_summary(self) -> None:
        """Print refinement summary."""
        total_tracks = sum(len(t) for t in self.trajectories.values())
        total_points = sum(
            len(traj.points) 
            for class_trajs in self.trajectories.values() 
            for traj in class_trajs.values()
        )
        total_outliers = sum(
            traj.outliers_removed
            for class_trajs in self.trajectories.values()
            for traj in class_trajs.values()
        )
        
        print(f"[TrackRefiner] Refined {total_tracks} tracks, {total_points} points")
        if total_outliers > 0:
            print(f"[TrackRefiner] Outliers corrected: {total_outliers}")
    
    def save_result(self, data: Dict[int, List[TrackedObject]], output_path: Path) -> None:
        """Save refined tracked objects."""
        serializable = {
            str(k): [t.model_dump() for t in v]
            for k, v in data.items()
        }
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def load_result(self, input_path: Path) -> Dict[int, List[TrackedObject]]:
        """Load refined tracked objects from file."""
        with open(input_path, 'r') as f:
            raw = json.load(f)
        return {
            int(k): [TrackedObject(**t) for t in v]
            for k, v in raw.items()
        }
