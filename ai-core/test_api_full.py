#!/usr/bin/env python3
"""
Full API Test Script with Interactive Disc Selection.

This script performs the complete testing workflow:
1. Verifies the API server is running
2. Launches the interactive disc selection tool
3. Uploads the video with fresh disc coordinates
4. Monitors processing progress
5. Displays results and optionally launches the viewer

IMPORTANT: Start the API server manually before running this script:
    cd ai-core
    PYTHONPATH=src:. uv run python run_api.py

Then run this script:
    cd ai-core
    PYTHONPATH=src:. uv run python test_api_full.py --video ../data/raw/video_test_1.mp4
    
Options:
    --skip-selection: Use existing selection coordinates
    --selection-file: Path to existing selection JSON
    --launch-viewer: Open interactive viewer after processing
"""

import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "src"))

import requests

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1/videos"
API_HEALTH_URL = "http://localhost:8000/health"
PROCESSING_POLL_INTERVAL = 3  # seconds
PROCESSING_TIMEOUT = 300  # seconds (5 minutes)


def check_server_running() -> bool:
    """Check if the API server is already running."""
    try:
        response = requests.get(API_HEALTH_URL, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def run_disc_selector(video_path: Path, output_path: Path) -> dict:
    """Run the interactive disc selector tool."""
    print(f"\n{'='*60}")
    print("  DISC SELECTION")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    print("\nInstructions:")
    print("  1. Click 'Centro' then click on the disc center")
    print("  2. Click 'Borde' then click on the disc edge")
    print("  3. Click 'Aceptar' to confirm")
    print(f"{'='*60}\n")
    
    # Import here to avoid issues if PyQt5/OpenCV not available
    from tools.disc_selector import DiscSelector
    
    result = DiscSelector.select_from_video(str(video_path))
    
    if not result.get("accepted"):
        print("[Selection] Selection was cancelled")
        sys.exit(1)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[Selection] Saved to: {output_path}")
    print(f"[Selection] Center: {result['center']}")
    print(f"[Selection] Radius: {result['radius']:.1f}px")
    
    return result


def upload_video(video_path: Path, selection: dict) -> str:
    """Upload video to API with disc selection."""
    print(f"\n[Upload] Uploading {video_path.name}...")
    
    center = selection["center"]
    radius = selection["radius"]
    
    with open(video_path, 'rb') as f:
        files = {'file': (video_path.name, f, 'video/mp4')}
        data = {
            'disc_center_x': center[0],
            'disc_center_y': center[1],
            'disc_radius': radius
        }
        
        response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
    
    if response.status_code != 200:
        print(f"[Upload] ERROR: {response.status_code} - {response.text}")
        sys.exit(1)
    
    result = response.json()
    video_id = result["video_id"]
    
    print(f"[Upload] Success! Video ID: {video_id}")
    print(f"[Upload] Disc selection: center=({center[0]}, {center[1]}), radius={radius:.0f}")
    
    return video_id


def wait_for_processing(video_id: str) -> dict:
    """Poll API until processing completes."""
    print(f"\n[Processing] Waiting for processing to complete...")
    print(f"[Processing] Video ID: {video_id}")
    
    start_time = time.time()
    last_status = None
    consecutive_404 = 0
    max_consecutive_404 = 10  # Allow some 404s before checking if processing finished
    
    # Give server a moment to register the job
    time.sleep(2)
    
    while time.time() - start_time < PROCESSING_TIMEOUT:
        try:
            response = requests.get(f"{API_BASE_URL}/{video_id}/status", timeout=5)
            
            if response.status_code == 200:
                consecutive_404 = 0
                data = response.json()
                status = data.get("status")
                
                if status != last_status:
                    elapsed = time.time() - start_time
                    progress = data.get("progress", 0) * 100
                    step = data.get("current_step", "")
                    print(f"[Processing] Status: {status} ({elapsed:.1f}s, {progress:.0f}%) {step}")
                    last_status = status
                
                if status == "completed":
                    print(f"[Processing] Complete!")
                    return data
                elif status == "failed":
                    print(f"[Processing] FAILED: {data.get('message', 'Unknown error')}")
                    sys.exit(1)
            elif response.status_code == 404:
                consecutive_404 += 1
                if consecutive_404 <= 3:
                    # Could be timing issue, wait and retry
                    print(f"[Processing] Job not found yet, waiting... ({consecutive_404})")
                elif consecutive_404 > max_consecutive_404:
                    # Check if results exist anyway (processing might have completed)
                    results_path = Path(f"../data/api/results/{video_id}/results.json")
                    if results_path.exists():
                        print(f"[Processing] Results file found, processing completed!")
                        return {"status": "completed", "video_id": video_id}
                    print(f"[Processing] Warning: Status check returned 404 ({consecutive_404}x)")
            else:
                print(f"[Processing] Warning: Status check returned {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"[Processing] Warning: Request failed: {e}")
        
        time.sleep(PROCESSING_POLL_INTERVAL)
    
    # Final check - maybe processing completed but status was lost
    results_path = Path(f"../data/api/results/{video_id}/results.json")
    if results_path.exists():
        print(f"[Processing] Results file found after timeout, processing completed!")
        return {"status": "completed", "video_id": video_id}
    
    print(f"[Processing] TIMEOUT after {PROCESSING_TIMEOUT}s")
    sys.exit(1)


def get_results(video_id: str) -> dict:
    """Fetch processing results."""
    print(f"\n[Results] Fetching results...")
    
    response = requests.get(f"{API_BASE_URL}/{video_id}/results")
    
    if response.status_code != 200:
        print(f"[Results] ERROR: {response.status_code} - {response.text}")
        sys.exit(1)
    
    results = response.json()
    
    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Video ID: {results.get('video_id')}")
    
    metadata = results.get('metadata', {})
    print(f"Resolution: {metadata.get('width')}x{metadata.get('height')}")
    print(f"FPS: {metadata.get('fps')}")
    print(f"Frames: {metadata.get('frame_count')}")
    
    tracks = results.get('tracks', [])
    print(f"\nTracks: {len(tracks)}")
    for track in tracks:
        frames = track.get('frames', {})
        frame_idxs = sorted([int(k) for k in frames.keys()])
        print(f"  - Track {track['track_id']} ({track['class_name']}): "
              f"frames {min(frame_idxs)}-{max(frame_idxs)} ({len(frame_idxs)} frames)")
    
    metrics = results.get('metrics', {})
    metric_frames = metrics.get('frames', [])
    print(f"\nMetrics:")
    print(f"  Frames with data: {len(metric_frames)}")
    if metric_frames:
        print(f"  Frame range: {min(metric_frames)} - {max(metric_frames)}")
        times = metrics.get('time_s', [])
        if times:
            print(f"  Time range: {min(times):.3f}s - {max(times):.3f}s")
    
    summary = results.get('summary', {})
    if summary:
        print(f"\nPhysical Metrics:")
        print(f"  Peak velocity: {summary.get('peak_velocity_m_s', 0):.2f} m/s")
        print(f"  Peak power: {summary.get('peak_power_w', 0):.1f} W")
        print(f"  Max height: {summary.get('max_height_m', 0):.2f} m")
    
    print(f"{'='*60}")
    
    return results


def save_results_locally(video_id: str, results: dict, output_dir: Path):
    """Save results to local file for viewer."""
    results_path = output_dir / f"api_results_{video_id}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to: {results_path}")
    return results_path


def main():
    parser = argparse.ArgumentParser(
        description="Full API test with interactive disc selection"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--skip-selection", "-s",
        action="store_true",
        help="Skip disc selection (use existing selection file)"
    )
    parser.add_argument(
        "--selection-file", "-f",
        type=str,
        default=None,
        help="Path to existing selection JSON file (for --skip-selection)"
    )
    parser.add_argument(
        "--selection-output", "-o",
        type=str,
        default=None,
        help="Path to save selection JSON"
    )
    parser.add_argument(
        "--launch-viewer",
        action="store_true",
        help="Launch interactive viewer after processing"
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    
    # Selection output path
    if args.selection_output:
        selection_output = Path(args.selection_output)
    else:
        selection_output = Path(f"/tmp/disc_selection_{video_path.stem}.json")
    
    print(f"\n{'='*60}")
    print("  FULL API TEST")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Selection output: {selection_output}")
    print(f"Skip selection: {args.skip_selection}")
    print(f"{'='*60}\n")
    
    try:
        # 1. Check server is running (don't start a new one - avoids race conditions)
        print("[Setup] Checking if API server is running...")
        if not check_server_running():
            print("[Setup] ERROR: API server is not running!")
            print("[Setup] Please start it manually in a separate terminal:")
            print("")
            print("    cd /Users/nicolas/Documents/app-mvp/ai-core")
            print("    PYTHONPATH=src:. uv run python run_api.py")
            print("")
            sys.exit(1)
        print("[Setup] API server is running ✓")
        
        # 2. Disc selection
        if args.skip_selection:
            if args.selection_file:
                selection_path = Path(args.selection_file)
            else:
                selection_path = selection_output
            
            if not selection_path.exists():
                print(f"ERROR: Selection file not found: {selection_path}")
                sys.exit(1)
            
            with open(selection_path, 'r') as f:
                selection = json.load(f)
            print(f"[Selection] Loaded from: {selection_path}")
            print(f"[Selection] Center: {selection['center']}")
            print(f"[Selection] Radius: {selection['radius']:.1f}px")
        else:
            selection = run_disc_selector(video_path, selection_output)
        
        # 3. Upload video
        video_id = upload_video(video_path, selection)
        
        # 4. Wait for processing
        wait_for_processing(video_id)
        
        # 5. Get results
        results = get_results(video_id)
        
        # 6. Save results locally
        results_dir = script_dir.parent / "data" / "api" / "results" / video_id
        if results_dir.exists():
            results_path = results_dir / "results.json"
            print(f"\n[Viewer] Results available at: {results_path}")
            print(f"[Viewer] To launch viewer:")
            print(f"         cd ai-core")
            print(f"         PYTHONPATH=src:. uv run python view_analysis.py {results_path}")
        
        # 7. Optional: Launch viewer
        if args.launch_viewer:
            print(f"\n[Viewer] Launching interactive viewer...")
            subprocess.run([
                "uv", "run", "python", "view_analysis.py", str(results_path)
            ], cwd=str(script_dir), env={**os.environ, "PYTHONPATH": "src:."})
        
        print(f"\n{'='*60}")
        print("  TEST COMPLETE")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n[Test] Interrupted by user")


if __name__ == "__main__":
    main()
