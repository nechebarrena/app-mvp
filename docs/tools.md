# Tooling Guide

This document describes the auxiliary tools and scripts developed to support the AI Core workflow.

## Metadata Scanner (`metadata_scanner.py`)

### Purpose
To standardize video ingestion, every video file in `data/raw/` must have a corresponding JSON "sidecar" file containing metadata (camera specs, technical info, user notes). This script automates the creation of these files.

### Usage
The script is located at `ai-core/src/input_layer/metadata_scanner.py`.

**Run via Python:**
```bash
# From project root
python ai-core/src/input_layer/metadata_scanner.py
```

### Behavior
1.  Scans `data/raw/` for `.mp4` files.
2.  For each video found:
    *   Checks if `video_name.json` exists.
    *   **If missing:**
        *   Extracts technical specs (Resolution, FPS, Frame Count) using OpenCV.
        *   Generates a new JSON file with these specs and default placeholders for manual fields.
    *   **If present:** Skips the file (does not overwrite existing notes).

### The Sidecar JSON Format
The generated JSON follows this schema:

```json
{
  "technical": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "frame_count": 600,
    "duration_seconds": 20.0
  },
  "camera_specs": {
    "device_model": "Unknown (Auto-generated)",
    "focal_length_mm": null,
    "sensor_width_mm": null
  },
  "context": {
    "user_notes": "",
    "tags": ["auto-generated"]
  }
}
```
**Workflow:** After dropping a video into `raw`, run the scanner, then manually edit the `.json` if you know specific details (like the phone model used).

