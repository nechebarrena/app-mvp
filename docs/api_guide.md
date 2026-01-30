# FastAPI Backend Guide

## Overview

The FastAPI backend provides REST endpoints for video analysis, designed for mobile app integration.

## Architecture

```
┌─────────────────┐                    ┌─────────────────────────────────┐
│   Mobile App    │                    │         FastAPI Server          │
│                 │                    │                                 │
│  ┌───────────┐  │   POST /upload     │  ┌─────────┐   ┌─────────────┐ │
│  │   Video   │──┼───────────────────►│  │ Storage │   │   Pipeline  │ │
│  │  Capture  │  │                    │  │ Manager │   │   Runner    │ │
│  └───────────┘  │   GET /status      │  └────┬────┘   └──────┬──────┘ │
│                 │◄───────────────────┼───────┴───────────────┘        │
│  ┌───────────┐  │                    │                                 │
│  │   Local   │  │   GET /results     │  ┌─────────────────────────┐   │
│  │  Render   │◄─┼───────────────────►│  │   Background Worker     │   │
│  └───────────┘  │   (JSON ~200KB)    │  │   (async processing)    │   │
│                 │                    │  └─────────────────────────┘   │
└─────────────────┘                    └─────────────────────────────────┘
```

## Endpoints

### Upload Video
```http
POST /api/v1/videos/upload
Content-Type: multipart/form-data

file: <video_file>
disc_center_x: <optional float>  # X coordinate of disc center in first frame (pixels)
disc_center_y: <optional float>  # Y coordinate of disc center in first frame (pixels)
disc_radius: <optional float>    # Disc radius in first frame (pixels)
```

**Note:** The disc selection parameters are highly recommended for better tracking accuracy. They enable single-object tracking heuristics that significantly improve disc detection, especially in the early frames of the video.

**Response:**
```json
{
    "video_id": "abc123def456",
    "status": "pending",
    "message": "Video uploaded successfully. Processing will start shortly. Disc selection: center=(587, 623), radius=74"
}
```

### Check Status
```http
GET /api/v1/videos/{video_id}/status
```

**Response:**
```json
{
    "video_id": "abc123def456",
    "status": "processing",
    "progress": 0.45,
    "current_step": "yolo_detection",
    "message": "Running yolo_detection...",
    "created_at": "2026-01-29T10:00:00",
    "updated_at": "2026-01-29T10:01:30"
}
```

**Status Values:**
| Status | Description |
|--------|-------------|
| `pending` | Waiting in queue |
| `processing` | Currently being analyzed |
| `completed` | Ready to retrieve results |
| `failed` | An error occurred |

### Get Results
```http
GET /api/v1/videos/{video_id}/results
```

**Response (when completed):**
```json
{
    "video_id": "abc123def456",
    "status": "completed",
    "metadata": {
        "fps": 30.0,
        "width": 1080,
        "height": 1920,
        "duration_s": 3.67,
        "total_frames": 110
    },
    "tracks": [
        {
            "track_id": 1,
            "class_name": "disco",
            "frames": {
                "0": {
                    "bbox": {"x1": 100, "y1": 200, "x2": 150, "y2": 250},
                    "mask": [[100,200], [110,205], ...],
                    "confidence": 0.92
                },
                ...
            },
            "trajectory": [[125, 225], [127, 223], ...]
        }
    ],
    "metrics": {
        "frames": [0, 1, 2, ...],
        "time_s": [0.0, 0.033, 0.067, ...],
        "height_m": [0.5, 0.52, 0.55, ...],
        "speed_m_s": [0.0, 0.6, 1.2, ...],
        "power_w": [0, 150, 320, ...],
        ...
    },
    "summary": {
        "peak_speed_m_s": 4.02,
        "peak_power_w": 3827,
        "max_height_m": 1.47,
        "min_height_m": 0.3,
        "lift_duration_s": 2.1,
        "total_frames": 110
    },
    "processed_at": "2026-01-29T10:02:00"
}
```

### Delete Video
```http
DELETE /api/v1/videos/{video_id}
```

**Response:**
```json
{
    "video_id": "abc123def456",
    "deleted": true,
    "message": "Video and all associated data have been deleted."
}
```

### List All Videos
```http
GET /api/v1/videos
```

**Response:**
```json
{
    "count": 3,
    "videos": [
        {"video_id": "abc123", "status": "completed", "progress": 1.0, ...},
        {"video_id": "def456", "status": "processing", "progress": 0.45, ...},
        ...
    ]
}
```

## Running the Server

### Development (with auto-reload)
```bash
cd ai-core
PYTHONPATH=src:. uv run python run_api.py --reload
```

### Production
```bash
cd ai-core
PYTHONPATH=src:. uv run python run_api.py --host 0.0.0.0 --port 8000
```

### Using uvicorn directly
```bash
cd ai-core
PYTHONPATH=src:. uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Interactive Documentation

Once the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Storage

### Directory Structure
```
data/
├── api/
│   ├── uploads/         # Uploaded videos
│   │   ├── abc123/
│   │   │   └── input.mp4
│   │   └── def456/
│   │       └── input.mp4
│   └── results/         # Processing results
│       ├── jobs.json    # Job state persistence
│       ├── abc123/
│       │   └── results.json
│       └── def456/
│           └── results.json
```

### Cleanup
Jobs and files persist until explicitly deleted. For automatic cleanup:
```python
from api.storage import get_storage
storage = get_storage()
storage.cleanup_old_jobs(max_age_hours=24)  # Delete jobs older than 24h
```

## Error Handling

All errors return JSON with this structure:
```json
{
    "detail": "Error message here"
}
```

| Status Code | Meaning |
|-------------|---------|
| 400 | Bad request (invalid file type, etc.) |
| 404 | Video not found |
| 413 | File too large (>100MB) |
| 500 | Server error / Processing failed |

## Configuration

### Limits
| Setting | Value |
|---------|-------|
| Max file size | 100MB |
| Allowed formats | MP4, MOV, AVI, MKV, WebM |
| Recommended resolution | 720p - 1080p |

### CORS
By default, CORS allows all origins (for development). In production, restrict to your mobile app's origin.

## Mobile App Integration

### Typical Flow
```python
import requests
import time

BASE_URL = "http://192.168.1.100:8000"  # Server IP

# 1. Upload video
with open("video.mp4", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/videos/upload",
        files={"file": f}
    )
video_id = response.json()["video_id"]

# 2. Poll for completion
while True:
    status = requests.get(f"{BASE_URL}/api/v1/videos/{video_id}/status").json()
    print(f"Progress: {status['progress']*100:.0f}%")
    
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        raise Exception(status["message"])
    
    time.sleep(2)

# 3. Get results
results = requests.get(f"{BASE_URL}/api/v1/videos/{video_id}/results").json()

# 4. Render overlays locally using results["tracks"]
# ...

# 5. Optionally delete
requests.delete(f"{BASE_URL}/api/v1/videos/{video_id}")
```

### Rendering on Mobile
The `tracks` array contains all information needed to render overlays:
- `bbox`: Bounding box coordinates per frame
- `mask`: Polygon points for segmentation (if available)
- `trajectory`: Center points for drawing motion path

The mobile app should:
1. Load the original video (already on device)
2. For each frame, lookup detections in `tracks[].frames[frame_idx]`
3. Draw bounding boxes, masks, and trajectories
4. Optionally show metrics from `metrics` series

## Health Check
```http
GET /health
```

Returns server health and job statistics:
```json
{
    "status": "healthy",
    "jobs": {
        "total": 10,
        "pending": 1,
        "processing": 2,
        "completed": 6,
        "failed": 1
    }
}
```
