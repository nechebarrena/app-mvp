# API Guide — Server ↔ Mobile App Contract

> **Version:** 2.0.0
> **Last Updated:** February 13, 2026
> **Status:** MVP Phase 1 — Remote Processing with Cutie VOS

---

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

## Workflow

| Step | Method | Endpoint | Description |
|------|--------|----------|-------------|
| 1 | `POST` | `/api/v1/videos/upload` | Upload video + disc selection |
| 2 | `GET` | `/api/v1/videos/{video_id}/status` | Poll processing progress |
| 3 | `GET` | `/api/v1/videos/{video_id}/results` | Retrieve analysis results |
| 4 | `DELETE` | `/api/v1/videos/{video_id}` | Delete video and results (optional) |

---

## Endpoints

### 1. Upload Video

```http
POST /api/v1/videos/upload
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | binary | **Yes** | Video file (MP4, MOV, AVI, MKV, WebM). Max 100MB |
| `disc_center_x` | float | **Yes*** | Disc center X in first frame (pixels) |
| `disc_center_y` | float | **Yes*** | Disc center Y in first frame (pixels) |
| `disc_radius` | float | **Yes*** | Disc radius in first frame (pixels) |
| `tracking_backend` | string | No | `"cutie"` or `"yolo"` (uses server default if omitted) |

*\*Required when server uses Cutie backend (recommended default).*

**Response:**
```json
{
    "video_id": "abc123def45",
    "status": "pending",
    "message": "Video uploaded successfully. Processing will start shortly."
}
```

### 2. Check Status

```http
GET /api/v1/videos/{video_id}/status
```

**Response:**
```json
{
    "video_id": "abc123def45",
    "status": "processing",
    "progress": 0.45,
    "current_step": "cutie_disc_tracking",
    "message": "Running cutie_disc_tracking (2/5)...",
    "created_at": "2026-02-13T21:38:26",
    "updated_at": "2026-02-13T21:38:50"
}
```

| Status | Description |
|--------|-------------|
| `pending` | Waiting in queue |
| `processing` | Currently being analyzed |
| `completed` | Ready to retrieve results |
| `failed` | An error occurred |

**Polling:** Check every 2–5 seconds while `pending` or `processing`.

### 3. Get Results

```http
GET /api/v1/videos/{video_id}/results
```

This is the **main contract**. The response structure below defines what the mobile app must parse.

---

## Results JSON Contract

### Guaranteed fields (always present)

```json
{
    "video_id": "abc123def45",
    "status": "completed",
    "processed_at": "2026-02-13T22:19:01",

    "metadata": { ... },
    "tracks": [ ... ],
    "metrics": { ... },
    "summary": { ... }
}
```

### `metadata` (required)

Video properties for synchronizing overlays with playback.

```json
{
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "duration_s": 25.87,
    "total_frames": 776
}
```

| Field | Type | Description |
|-------|------|-------------|
| `fps` | float | Frames per second |
| `width` | int | Video width in pixels |
| `height` | int | Video height in pixels |
| `duration_s` | float | Duration in seconds |
| `total_frames` | int | Total frame count |

### `tracks` (required — at least the disc track)

Array of tracked objects. **The disc track is always present.** Person track is optional.

```json
[
    {
        "track_id": 1,
        "class_name": "frisbee",
        "trajectory": [[cx, cy], [cx, cy], ...],
        "frames": {
            "0": {
                "mask": [[x,y], [x,y], ...],
                "confidence": 0.99,
                "bbox": null
            },
            "1": { ... },
            ...
        }
    }
]
```

#### Track fields

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | int | Unique track identifier |
| `class_name` | string | `"frisbee"` = disc **(required)**, `"person"` = athlete (optional) |
| `trajectory` | `[[x,y], ...]` | Center points in pixels, one per detected frame |
| `frames` | `{frame_idx: detection}` | Per-frame detection data (keys are string frame indices) |

#### Frame detection fields

| Field | Type | Disc | Person | Description |
|-------|------|------|--------|-------------|
| `mask` | `[[x,y], ...]` or `null` | **Required** | Optional | Polygon contour points in pixels |
| `confidence` | float | **Required** | **Required** | Detection confidence 0.0–1.0 |
| `bbox` | `{x1,y1,x2,y2}` or `null` | Optional | Optional | Bounding box in pixels |

#### Class name mapping

| `class_name` value | Semantic meaning | Notes |
|---------------------|-----------------|-------|
| `"frisbee"` | Weightlifting disc | COCO legacy name. Always present. |
| `"person"` | Athlete | Only present if person detection is enabled server-side |

### `metrics` (required)

Time series for the disc. All arrays have the same length and are aligned by index.

```json
{
    "frames":             [0, 1, 2, ...],
    "time_s":             [0.0, 0.033, 0.067, ...],
    "x_m":                [0.5, 0.51, ...],
    "y_m":                [1.2, 1.19, ...],
    "height_m":           [0.0, 0.02, ...],
    "vx_m_s":             [0.1, 0.12, ...],
    "vy_m_s":             [0.5, 0.6, ...],
    "speed_m_s":          [0.51, 0.61, ...],
    "accel_m_s2":         [0.0, 2.5, ...],
    "kinetic_energy_j":   [2.6, 3.7, ...],
    "potential_energy_j":  [0.0, 3.9, ...],
    "total_energy_j":     [2.6, 7.6, ...],
    "power_w":            [0.0, 150, ...]
}
```

| Series | Unit | Description |
|--------|------|-------------|
| `frames` | index | Frame number (integer) |
| `time_s` | seconds | Timestamp |
| `x_m` | meters | Horizontal position |
| `y_m` | meters | Vertical position |
| `height_m` | meters | Height above lowest point |
| `vx_m_s` | m/s | Horizontal velocity |
| `vy_m_s` | m/s | Vertical velocity |
| `speed_m_s` | m/s | Total speed (magnitude) |
| `accel_m_s2` | m/s² | Total acceleration |
| `kinetic_energy_j` | Joules | Kinetic energy |
| `potential_energy_j` | Joules | Gravitational potential energy |
| `total_energy_j` | Joules | Total mechanical energy |
| `power_w` | Watts | Instantaneous power |

### `summary` (required)

Peak values for quick display.

```json
{
    "peak_speed_m_s": 4.02,
    "peak_power_w": 3827.0,
    "max_height_m": 1.47,
    "min_height_m": 0.30,
    "lift_duration_s": 2.1,
    "total_frames": 110
}
```

---

## What the mobile app MUST handle

1. **`metadata`** — to sync video playback with frame indices (`fps`, resolution)
2. **Disc track** (`class_name == "frisbee"`) — always present:
   - `mask` per frame → draw disc contour overlay on video
   - `trajectory` → draw the full motion path
   - `confidence` per frame → optionally show detection quality
3. **`metrics`** — all 13 series for graphs/charts
4. **`summary`** — peak values for results screen

## What the mobile app SHOULD handle gracefully

- **Person track** (`class_name == "person"`) — may or may not be present
- **`bbox`** in any frame detection — may be `null`
- **Additional tracks** — ignore any `class_name` not recognized
- **Empty metrics** — if pipeline fails to compute metrics, arrays may be empty

## What is NOT in the contract (currently)

- **Pose keypoints** — server can run pose estimation but keypoints are not included in results
- **Physical calibration parameters** — disc diameter, weight are server-side only
- **Raw detection scores per model** — only final confidence is exposed

---

## Server Configuration Endpoints

These are used by the Control Panel, not by the mobile app.

```http
GET  /api/v1/config/models         # Current model config
POST /api/v1/config/models         # Set model config
GET  /api/v1/config/tracking-backend
POST /api/v1/config/tracking-backend
```

---

## Running the Server

### Via Control Panel (recommended)
```bash
cd ai-core
PYTHONPATH=vendors/cutie:src:. uv run python control_panel.py
```

### Direct
```bash
cd ai-core
PYTHONPATH=vendors/cutie:src:. uv run python run_api.py
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Error Handling

```json
{"detail": "Error message here"}
```

| Status Code | Meaning |
|-------------|---------|
| 400 | Bad request (invalid file type) |
| 404 | Video not found |
| 413 | File too large (>100MB) |
| 500 | Processing failed |

---

## Limits

| Setting | Value |
|---------|-------|
| Max file size | 100MB |
| Allowed formats | MP4, MOV, AVI, MKV, WebM |
| Recommended resolution | 720p – 1080p |
| Max concurrent processing | 1 (MVP single-user) |

---

## Benchmark & Testing Endpoints

Three additional endpoints are available for external test/benchmark tooling. They do **not** affect the mobile workflow.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/info` | Server capabilities handshake |
| `POST` | `/api/v1/bench/run_one` | Submit one benchmark case (upload or local_asset) |
| `GET` | `/api/v1/assets` | List local-asset videos in `DATASETS_ROOT` |

### `GET /api/v1/info`

Returns `api_version`, `results_contract_version`, `available_backends`, `current_default_backend`, and server limits. Recommended first call for any external tool.

```bash
curl -s http://localhost:8000/api/v1/info | python3 -m json.tool
```

### `POST /api/v1/bench/run_one`

Like `/videos/upload` but adds trazabilidad fields (`case_id`, `client_run_id`, `tags`) and supports `video_source_type=local_asset` (video already on server). Returns the same `job_id` — poll with `/videos/{id}/status` as usual.

```bash
# Upload mode
curl -X POST http://localhost:8000/api/v1/bench/run_one \
  -F video_source_type=upload \
  -F "file=@/path/to/video.mp4" \
  -F case_id=snatch_001 \
  -F client_run_id=run_20260213

# Local-asset mode (video already in data/bench_assets/)
curl -X POST http://localhost:8000/api/v1/bench/run_one \
  -F video_source_type=local_asset \
  -F asset_id=snatch_001 \
  -F case_id=snatch_001
```

See `docs/BENCHMARK_API.md` for full field reference and curl examples.
