#!/usr/bin/env python3
"""
Interactive Analysis Viewer Launcher

This script launches the interactive visualization tool for analyzing
lifting metrics after the pipeline has completed.

Usage:
    PYTHONPATH=src:. uv run python view_analysis.py [run_dir]

Arguments:
    run_dir: Directory containing pipeline outputs (default: full_analysis_run)

Example:
    PYTHONPATH=src:. uv run python view_analysis.py full_analysis_run
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visualization.interactive_viewer import launch_interactive_viewer


def find_files(run_dir: Path):
    """Find the required files in the run directory."""
    # Look for metrics CSV
    metrics_csv = None
    for pattern in ["metrics_calculator_output.csv", "*metrics*.csv"]:
        files = list(run_dir.glob(pattern))
        if files:
            metrics_csv = files[0]
            break
    
    # Look for tracking video
    video_mp4 = None
    for pattern in ["tracking_video.mp4", "*tracking*.mp4", "*.mp4"]:
        files = list(run_dir.glob(pattern))
        if files:
            video_mp4 = files[0]
            break
    
    return metrics_csv, video_mp4


def main():
    parser = argparse.ArgumentParser(
        description="Launch interactive analysis viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="full_analysis_run",
        help="Directory containing pipeline outputs (default: full_analysis_run)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Path to metrics CSV file (optional, auto-detected)"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file (optional, auto-detected)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=29.5,
        help="Video FPS (default: 29.5)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent / "data" / "outputs"
    run_dir = base_dir / args.run_dir
    
    if not run_dir.exists():
        # Try as absolute path
        run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        print(f"Available directories in {base_dir}:")
        if base_dir.exists():
            for d in base_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        sys.exit(1)
    
    print(f"[Viewer] Loading from: {run_dir}")
    
    # Find files
    if args.metrics:
        metrics_csv = Path(args.metrics)
    else:
        metrics_csv, _ = find_files(run_dir)
    
    if args.video:
        video_mp4 = Path(args.video)
    else:
        _, video_mp4 = find_files(run_dir)
    
    # Validate files
    if metrics_csv is None or not metrics_csv.exists():
        print(f"Error: Metrics CSV not found in {run_dir}")
        print("Files in directory:")
        for f in run_dir.iterdir():
            print(f"  - {f.name}")
        sys.exit(1)
    
    if video_mp4 is None or not video_mp4.exists():
        print(f"Error: Video file not found in {run_dir}")
        print("Files in directory:")
        for f in run_dir.iterdir():
            print(f"  - {f.name}")
        sys.exit(1)
    
    print(f"[Viewer] Metrics: {metrics_csv.name}")
    print(f"[Viewer] Video: {video_mp4.name}")
    
    # Load metrics
    metrics_df = pd.read_csv(metrics_csv)
    print(f"[Viewer] Loaded {len(metrics_df)} data points")
    
    # Summary
    if 'speed_m_s' in metrics_df.columns:
        print(f"[Viewer] Peak speed: {metrics_df['speed_m_s'].max():.2f} m/s")
    if 'power_w' in metrics_df.columns:
        print(f"[Viewer] Peak power: {metrics_df['power_w'].max():.0f} W")
    if 'height_m' in metrics_df.columns:
        print(f"[Viewer] Max height: {metrics_df['height_m'].max():.2f} m")
    
    print("\n[Viewer] Opening interactive viewer...")
    print("[Viewer] Use the dropdowns to select different graphs")
    print("[Viewer] Use video controls to navigate")
    print("[Viewer] Close button or window X to exit\n")
    
    # Launch viewer
    launch_interactive_viewer(
        str(video_mp4),
        metrics_df,
        args.fps,
        f"Análisis de Levantamiento - {run_dir.name}"
    )


if __name__ == "__main__":
    main()
