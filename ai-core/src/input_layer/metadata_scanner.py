import json
import cv2
from pathlib import Path
from typing import Dict, Any

def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Extracts technical metadata from a video file using OpenCV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": round(duration, 2)
    }

def generate_sidecar_json(video_path: Path, force: bool = False) -> None:
    """
    Generates a .json sidecar file for a given video path.
    If the json exists, it skips unless force=True.
    """
    json_path = video_path.with_suffix('.json')
    
    if json_path.exists() and not force:
        print(f"Skipping {video_path.name}: Metadata file already exists.")
        return

    try:
        specs = extract_video_metadata(video_path)
    except Exception as e:
        print(f"Error processing {video_path.name}: {e}")
        return

    # Structure aligning with CameraSpecs + User/Business Metadata
    metadata = {
        # Technical Specs (Auto-extracted)
        "technical": {
            "width": specs["width"],
            "height": specs["height"],
            "fps": specs["fps"],
            "frame_count": specs["frame_count"],
            "duration_seconds": specs["duration_seconds"]
        },
        # Semantic Specs (To be filled by user or defaults)
        "camera_specs": {
            "device_model": "Unknown (Auto-generated)",
            "focal_length_mm": None,
            "sensor_width_mm": None
        },
        "context": {
            "user_notes": "",
            "tags": ["auto-generated"]
        }
    }

    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated metadata for {video_path.name} -> {json_path.name}")

def scan_directory(directory: Path) -> None:
    """Scans a directory for mp4 files and ensures they have metadata."""
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    print(f"Scanning {directory} for videos...")
    video_files = list(directory.glob("*.mp4"))
    
    if not video_files:
        print("No .mp4 files found.")
        return

    for video in video_files:
        generate_sidecar_json(video)

if __name__ == "__main__":
    # Default behavior: scan data/raw relative to project root
    # Assuming script is run from project root or ai-core/src/input_layer/
    
    # Try to find project root by looking for 'data' folder up the tree
    current_path = Path(__file__).resolve()
    project_root = None
    
    for parent in current_path.parents:
        if (parent / "data" / "raw").exists():
            project_root = parent
            break
            
    if project_root:
        raw_data_dir = project_root / "data" / "raw"
        scan_directory(raw_data_dir)
    else:
        print("Could not locate data/raw directory. Please run from project root.")

