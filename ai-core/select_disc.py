#!/usr/bin/env python
"""
Script to run the disc selector tool interactively.

Usage:
    python select_disc.py                           # Uses default video
    python select_disc.py path/to/video.mp4         # Specify video
    python select_disc.py video.mp4 output.json     # Specify output file

The selection is saved to a JSON file that can be used by the pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tools.disc_selector import DiscSelector, save_selection


def main():
    # Default paths
    default_video = Path(__file__).parent.parent / "data" / "raw" / "video_test_1.mp4"
    default_output = Path(__file__).parent.parent / "data" / "outputs" / "disc_selection.json"
    
    # Parse arguments
    video_path = sys.argv[1] if len(sys.argv) > 1 else str(default_video)
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(default_output)
    
    print(f"[DiscSelector] Video: {video_path}")
    print(f"[DiscSelector] Output: {output_path}")
    print()
    print("Instructions:")
    print("  1. Click 'Centro' button, then click on the disc center")
    print("  2. Click 'Borde' button, then click on the disc edge")
    print("  3. Click 'Aceptar' to save or 'Resetear' to start over")
    print("  4. Press ESC to cancel")
    print()
    
    # Run selector
    result = DiscSelector.select_from_video(video_path)
    
    if result["accepted"]:
        save_selection(result, output_path)
        print()
        print(f"Result: center=({result['center'][0]}, {result['center'][1]}), radius={result['radius']:.1f}")
    else:
        print("Selection cancelled.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
