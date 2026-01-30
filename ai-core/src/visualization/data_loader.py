"""
Unified Data Loader for Interactive Viewer.

Supports loading analysis data from multiple sources:
1. Pipeline output (CSV + video with overlays)
2. API JSON output (results.json + original video)

Both sources are converted to a common format for the InteractiveViewer.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ViewerData:
    """Data package for the InteractiveViewer."""
    video_path: str
    metrics_df: pd.DataFrame
    fps: float
    metadata: Dict[str, Any]
    source_type: str  # "pipeline" or "api"
    
    def __post_init__(self):
        """Validate the data."""
        if self.metrics_df.empty:
            raise ValueError("metrics_df cannot be empty")
        
        required_columns = ['frame_idx', 'time_s']
        missing = [c for c in required_columns if c not in self.metrics_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def load_from_pipeline(
    metrics_csv_path: str,
    video_path: str,
    fps: Optional[float] = None
) -> ViewerData:
    """
    Load data from pipeline output.
    
    Args:
        metrics_csv_path: Path to metrics CSV file (from MetricsCalculator)
        video_path: Path to video file (tracking video with overlays)
        fps: Frame rate (optional, will try to read from video)
        
    Returns:
        ViewerData ready for InteractiveViewer
    """
    metrics_path = Path(metrics_csv_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv_path}")
    
    video_path = str(Path(video_path).resolve())
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Load metrics
    metrics_df = pd.read_csv(metrics_path)
    
    # Get FPS from video if not provided
    if fps is None:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            fps = 30.0  # Default fallback
    
    # Build metadata
    metadata = {
        "source": "pipeline",
        "metrics_path": str(metrics_path),
        "video_path": video_path,
        "total_frames": len(metrics_df),
    }
    
    return ViewerData(
        video_path=video_path,
        metrics_df=metrics_df,
        fps=fps,
        metadata=metadata,
        source_type="pipeline"
    )


def load_from_api_json(
    results_json_path: str,
    video_path: Optional[str] = None
) -> ViewerData:
    """
    Load data from API JSON output.
    
    Args:
        results_json_path: Path to API results.json file
        video_path: Path to video file (optional, will try to find in API uploads)
        
    Returns:
        ViewerData ready for InteractiveViewer
    """
    json_path = Path(results_json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_json_path}")
    
    # Load JSON
    with open(json_path) as f:
        data = json.load(f)
    
    # Extract video_id and find video
    video_id = data.get("video_id", "")
    
    if video_path is None:
        # Try to find video in standard API locations
        api_base = json_path.parent.parent  # results/{video_id}/ -> results/
        possible_paths = [
            api_base.parent / "uploads" / video_id / "input.mp4",  # API upload location
            api_base.parent.parent / "raw" / f"{video_id}.mp4",    # data/raw/ symlink
        ]
        
        for p in possible_paths:
            if p.exists():
                video_path = str(p.resolve())
                break
        
        if video_path is None:
            raise FileNotFoundError(
                f"Video not found for {video_id}. "
                f"Searched: {[str(p) for p in possible_paths]}. "
                f"Please provide video_path explicitly."
            )
    else:
        video_path = str(Path(video_path).resolve())
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Convert metrics to DataFrame
    metrics = data.get("metrics", {})
    if not metrics or not metrics.get("frames"):
        raise ValueError("No metrics data in JSON")
    
    metrics_df = pd.DataFrame({
        "frame_idx": metrics.get("frames", []),
        "time_s": metrics.get("time_s", []),
        "x_m": metrics.get("x_m", []),
        "y_m": metrics.get("y_m", []),
        "height_m": metrics.get("height_m", []),
        "vx_m_s": metrics.get("vx_m_s", []),
        "vy_m_s": metrics.get("vy_m_s", []),
        "speed_m_s": metrics.get("speed_m_s", []),
        "accel_m_s2": metrics.get("accel_m_s2", []),
        "kinetic_energy_j": metrics.get("kinetic_energy_j", []),
        "potential_energy_j": metrics.get("potential_energy_j", []),
        "total_energy_j": metrics.get("total_energy_j", []),
        "power_w": metrics.get("power_w", []),
    })
    
    # Remove columns that are all empty
    metrics_df = metrics_df.dropna(axis=1, how='all')
    
    # Get FPS from metadata
    api_metadata = data.get("metadata", {})
    fps = api_metadata.get("fps", 30.0)
    
    # Build metadata
    metadata = {
        "source": "api",
        "video_id": video_id,
        "json_path": str(json_path),
        "video_path": video_path,
        "api_metadata": api_metadata,
        "summary": data.get("summary", {}),
        "tracks_count": len(data.get("tracks", [])),
    }
    
    return ViewerData(
        video_path=video_path,
        metrics_df=metrics_df,
        fps=fps,
        metadata=metadata,
        source_type="api"
    )


def load_viewer_data(
    source: str,
    video_path: Optional[str] = None,
    **kwargs
) -> ViewerData:
    """
    Auto-detect source type and load data.
    
    Args:
        source: Path to either:
            - CSV file (pipeline mode)
            - JSON file (API mode)
            - Directory containing results
        video_path: Path to video (required for pipeline, optional for API)
        **kwargs: Additional arguments passed to specific loaders
        
    Returns:
        ViewerData ready for InteractiveViewer
    """
    source_path = Path(source)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    
    # Auto-detect source type
    if source_path.is_dir():
        # Directory - look for results.json or metrics CSV
        json_file = source_path / "results.json"
        csv_file = source_path / "metrics_calculator_output.csv"
        
        if json_file.exists():
            return load_from_api_json(str(json_file), video_path)
        elif csv_file.exists():
            if video_path is None:
                # Try to find tracking video
                tracking_video = source_path / "tracking_video.mp4"
                if tracking_video.exists():
                    video_path = str(tracking_video)
                else:
                    raise ValueError("video_path required for pipeline mode")
            return load_from_pipeline(str(csv_file), video_path, kwargs.get("fps"))
        else:
            raise ValueError(f"No results.json or metrics CSV found in {source}")
    
    elif source_path.suffix.lower() == ".json":
        return load_from_api_json(str(source_path), video_path)
    
    elif source_path.suffix.lower() == ".csv":
        if video_path is None:
            raise ValueError("video_path required for CSV input")
        return load_from_pipeline(str(source_path), video_path, kwargs.get("fps"))
    
    else:
        raise ValueError(f"Unknown source type: {source}")


def print_data_summary(data: ViewerData) -> None:
    """Print a summary of loaded data."""
    print(f"\n{'='*60}")
    print(f"  Viewer Data Summary")
    print(f"{'='*60}")
    print(f"  Source: {data.source_type}")
    print(f"  Video: {data.video_path}")
    print(f"  FPS: {data.fps:.2f}")
    print(f"  Frames in metrics: {len(data.metrics_df)}")
    print(f"  Columns: {list(data.metrics_df.columns)}")
    
    if data.source_type == "api":
        summary = data.metadata.get("summary", {})
        print(f"\n  API Summary:")
        print(f"    Peak Speed: {summary.get('peak_speed_m_s', 0):.2f} m/s")
        print(f"    Peak Power: {summary.get('peak_power_w', 0):.0f} W")
        print(f"    Tracks: {data.metadata.get('tracks_count', 0)}")
    
    print(f"{'='*60}\n")
