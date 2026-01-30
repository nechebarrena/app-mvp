#!/usr/bin/env python
"""
Launch Interactive Analysis Viewer.

Supports multiple input sources:
1. Pipeline output directory (with CSV metrics + tracking video)
2. API results.json (with original video)

Usage:
    # From pipeline output
    cd ai-core
    PYTHONPATH=src:. uv run python view_analysis.py ../data/outputs/full_analysis_run
    
    # From API results
    PYTHONPATH=src:. uv run python view_analysis.py ../data/api/results/abc123/results.json
    
    # With explicit video path
    PYTHONPATH=src:. uv run python view_analysis.py results.json --video /path/to/video.mp4
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visualization.data_loader import load_viewer_data, print_data_summary
from visualization.interactive_viewer import launch_interactive_viewer


def main():
    parser = argparse.ArgumentParser(
        description="Launch Interactive Analysis Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From pipeline output directory
  %(prog)s ../data/outputs/full_analysis_run
  
  # From API results.json
  %(prog)s ../data/api/results/abc123/results.json
  
  # With explicit video path
  %(prog)s results.json --video /path/to/video.mp4
  
  # From metrics CSV (pipeline)
  %(prog)s metrics.csv --video tracking_video.mp4
"""
    )
    
    parser.add_argument(
        "source",
        help="Path to results directory, results.json, or metrics CSV"
    )
    
    parser.add_argument(
        "--video", "-v",
        help="Path to video file (optional for API, required for CSV)"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        help="Video FPS (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--title", "-t",
        default="Análisis de Levantamiento",
        help="Window title"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Don't print data summary"
    )
    
    args = parser.parse_args()
    
    # Resolve source path
    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = Path(__file__).parent / args.source
    
    if not source_path.exists():
        print(f"Error: Source not found: {source_path}")
        sys.exit(1)
    
    # Resolve video path if provided
    video_path = None
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = Path(__file__).parent / args.video
        video_path = str(video_path)
    
    try:
        # Load data
        print(f"Loading data from: {source_path}")
        data = load_viewer_data(
            str(source_path),
            video_path=video_path,
            fps=args.fps
        )
        
        if not args.quiet:
            print_data_summary(data)
        
        # Launch viewer
        print("Launching interactive viewer...")
        print("(Close window to exit)")
        
        launch_interactive_viewer(
            video_path=data.video_path,
            metrics_df=data.metrics_df,
            fps=data.fps,
            title=f"{args.title} ({data.source_type.upper()})"
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
