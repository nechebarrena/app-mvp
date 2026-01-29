#!/usr/bin/env python
"""
Full API Test Script with Disc Selection.

This script simulates the complete mobile app flow:
1. Run disc selection tool (GUI)
2. Upload video + selection to API
3. Poll for completion
4. Display results

Usage:
    cd ai-core
    PYTHONPATH=src:. uv run python test_api_full.py [video_path]
    
    Default video: ../data/raw/video_test_1.mp4
"""

import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not installed. Run: uv pip install requests")
    sys.exit(1)


# Default configuration
DEFAULT_VIDEO = "../data/raw/video_test_1.mp4"
API_BASE_URL = "http://localhost:8000"
SELECTION_OUTPUT = "../data/outputs/api_disc_selection.json"


def run_disc_selection(video_path: str, output_file: str) -> dict:
    """
    Run the disc selection tool and return the selection data.
    """
    print("\n" + "="*60)
    print("  STEP 1: Disc Selection")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Output: {output_file}")
    print("\nOpening selection tool...")
    print("  - Click 'Centro' then click on disc center")
    print("  - Click 'Borde' then click on disc edge")
    print("  - Click 'Aceptar' to save")
    print()
    
    # Run the selection tool
    result = subprocess.run(
        [
            sys.executable, "select_disc.py",
            "--video", video_path,
            "--output", output_file
        ],
        cwd=Path(__file__).parent
    )
    
    if result.returncode != 0:
        print("Error: Selection tool failed or was cancelled")
        sys.exit(1)
    
    # Load the selection data
    output_path = Path(__file__).parent / output_file
    if not output_path.exists():
        print(f"Error: Selection file not created: {output_path}")
        sys.exit(1)
    
    with open(output_path) as f:
        selection_data = json.load(f)
    
    print(f"\n✓ Selection saved:")
    print(f"  Center: ({selection_data['center'][0]:.1f}, {selection_data['center'][1]:.1f})")
    print(f"  Radius: {selection_data['radius']:.1f} px")
    
    return selection_data


def check_api_available() -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def upload_video(video_path: str, selection_data: dict) -> str:
    """
    Upload video with selection data to the API.
    Returns video_id.
    """
    print("\n" + "="*60)
    print("  STEP 2: Upload Video to API")
    print("="*60)
    
    if not check_api_available():
        print(f"\nError: API server not running at {API_BASE_URL}")
        print("Start it with:")
        print("  cd ai-core")
        print("  PYTHONPATH=src:. uv run python run_api.py")
        sys.exit(1)
    
    print(f"API: {API_BASE_URL}")
    print(f"Video: {video_path}")
    print(f"Selection: center={selection_data['center']}, radius={selection_data['radius']:.1f}")
    print("\nUploading...")
    
    with open(video_path, 'rb') as f:
        files = {'file': (Path(video_path).name, f, 'video/mp4')}
        data = {
            'disc_center_x': selection_data['center'][0],
            'disc_center_y': selection_data['center'][1],
            'disc_radius': selection_data['radius']
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/videos/upload",
            files=files,
            data=data
        )
    
    if response.status_code != 200:
        print(f"Error: Upload failed with status {response.status_code}")
        print(response.text)
        sys.exit(1)
    
    result = response.json()
    video_id = result['video_id']
    
    print(f"\n✓ Upload successful")
    print(f"  Video ID: {video_id}")
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")
    
    return video_id


def poll_status(video_id: str, timeout: int = 120) -> dict:
    """
    Poll the API for processing status until complete or failed.
    """
    print("\n" + "="*60)
    print("  STEP 3: Wait for Processing")
    print("="*60)
    
    start_time = time.time()
    last_progress = -1
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"\nError: Timeout after {timeout}s")
            sys.exit(1)
        
        response = requests.get(f"{API_BASE_URL}/api/v1/videos/{video_id}/status")
        if response.status_code != 200:
            print(f"Error: Status check failed: {response.text}")
            sys.exit(1)
        
        status = response.json()
        progress = status.get('progress', 0)
        current_step = status.get('current_step', '-')
        state = status['status']
        
        # Only print when progress changes
        if int(progress * 100) != last_progress:
            last_progress = int(progress * 100)
            print(f"  [{elapsed:5.1f}s] {state:12} | {last_progress:3}% | {current_step}")
        
        if state == 'completed':
            print(f"\n✓ Processing completed in {elapsed:.1f}s")
            return status
        elif state == 'failed':
            print(f"\n✗ Processing failed: {status.get('message')}")
            sys.exit(1)
        
        time.sleep(2)


def get_results(video_id: str) -> dict:
    """
    Get the analysis results.
    """
    print("\n" + "="*60)
    print("  STEP 4: Results")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/api/v1/videos/{video_id}/results")
    if response.status_code != 200:
        print(f"Error: Failed to get results: {response.text}")
        sys.exit(1)
    
    results = response.json()
    
    print(f"\n=== Video Metadata ===")
    meta = results.get('metadata', {})
    print(f"  Resolution: {meta.get('width')}x{meta.get('height')}")
    print(f"  FPS: {meta.get('fps'):.2f}")
    print(f"  Duration: {meta.get('duration_s'):.2f}s ({meta.get('total_frames')} frames)")
    
    print(f"\n=== Tracks ===")
    tracks = results.get('tracks', [])
    print(f"  Total tracks: {len(tracks)}")
    for t in tracks:
        print(f"  - Track {t['track_id']}: {t['class_name']} ({len(t['frames'])} frames, {len(t['trajectory'])} trajectory points)")
    
    print(f"\n=== Metrics Summary ===")
    summary = results.get('summary', {})
    print(f"  Peak Speed: {summary.get('peak_speed_m_s', 0):.2f} m/s")
    print(f"  Peak Power: {summary.get('peak_power_w', 0):.0f} W")
    print(f"  Max Height: {summary.get('max_height_m', 0):.2f} m")
    print(f"  Min Height: {summary.get('min_height_m', 0):.2f} m")
    
    # Save results to file
    results_file = Path(__file__).parent / f"api_results_{video_id}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Full API test with disc selection"
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=f"Path to video file (default: {DEFAULT_VIDEO})"
    )
    parser.add_argument(
        "--selection",
        default=SELECTION_OUTPUT,
        help="Output file for disc selection"
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Skip selection tool and use existing selection file"
    )
    parser.add_argument(
        "--api-url",
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})"
    )
    
    args = parser.parse_args()
    
    # Resolve video path
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = Path(__file__).parent / args.video
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    global API_BASE_URL
    API_BASE_URL = args.api_url
    
    print("\n" + "="*60)
    print("  API FULL TEST - Disc Selection + Processing")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"API: {API_BASE_URL}")
    
    # Step 1: Disc Selection
    selection_file = Path(__file__).parent / args.selection
    if args.skip_selection:
        if not selection_file.exists():
            print(f"Error: Selection file not found: {selection_file}")
            print("Run without --skip-selection first")
            sys.exit(1)
        with open(selection_file) as f:
            selection_data = json.load(f)
        print(f"\nUsing existing selection: {selection_file}")
        print(f"  Center: ({selection_data['center'][0]:.1f}, {selection_data['center'][1]:.1f})")
        print(f"  Radius: {selection_data['radius']:.1f} px")
    else:
        selection_data = run_disc_selection(str(video_path), args.selection)
    
    # Step 2: Upload
    video_id = upload_video(str(video_path), selection_data)
    
    # Step 3: Poll status
    poll_status(video_id)
    
    # Step 4: Get results
    results = get_results(video_id)
    
    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60)
    print(f"\nVideo ID: {video_id}")
    print(f"Tracks: {len(results.get('tracks', []))}")
    
    # Count frisbee tracks
    frisbee_tracks = [t for t in results.get('tracks', []) if t['class_name'] == 'frisbee']
    print(f"Frisbee tracks: {len(frisbee_tracks)}")
    
    if len(frisbee_tracks) == 1:
        print("\n✓ Single disc tracking PASSED - heuristics working correctly")
    elif len(frisbee_tracks) == 0:
        print("\n⚠ No frisbee tracks found - check detection")
    else:
        print(f"\n⚠ Multiple frisbee tracks ({len(frisbee_tracks)}) - heuristics may not be applied")


if __name__ == "__main__":
    main()
