# BENCHMARK_API.md — Living Documentation

> **Status:** v1.0 — implemented (February 2026)
> **Scope:** Full benchmark API surface + design notes
> **Contract version:** ResultsContract v2.0.0

---

## A. Executive Summary

The server exposes a REST API (FastAPI) that accepts a weightlifting video + an initial disc-selection seed, runs an AI pipeline (tracking + physical metrics) in the background, and returns a JSON with tracks, time series, and peak values.

The **benchmark extension** (this document) adds:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/info` | Stable handshake — server capabilities for any external client |
| `POST /api/v1/bench/run_one` | Submit one benchmark case (upload or local_asset) |
| `GET /api/v1/assets` | List local-asset videos available in `DATASETS_ROOT` |

Plus internal improvements:
- Each job now writes **`job_meta.json`** with full trazabilidad (timestamps, backend used, video metadata, `case_id`/`client_run_id`/`tags`).
- **`pipeline.log`** is now populated per-job (fixed `logging.basicConfig` one-shot issue).
- `VideoJob` tracks `started_at`, `finished_at`, `source_type`, `asset_id`, `case_id`, `client_run_id`, `tags`.

**Design principle:** the benchmark runner orchestrates the dataset and fires N independent calls — one video per request. The server sees only independent jobs and does not know about "runs" or "datasets".

---

## B. Full Endpoint Table

| Method | Path | Description | Breaking |
|--------|------|-------------|---------|
| `POST` | `/api/v1/videos/upload` | Upload video + seed → job | no |
| `GET` | `/api/v1/videos/{id}/status` | Poll job status/progress | no |
| `GET` | `/api/v1/videos/{id}/results` | Download ResultsContract JSON | no |
| `DELETE` | `/api/v1/videos/{id}` | Delete video + artefacts | no |
| `GET` | `/api/v1/videos` | List all jobs | no |
| `GET` | `/api/v1/config/models` | Current model config | no |
| `POST` | `/api/v1/config/models` | Set backend + optional models | no |
| `GET` | `/api/v1/config/tracking-backend` | Active backend | no |
| `POST` | `/api/v1/config/tracking-backend` | Change backend | no |
| **`GET`** | **`/api/v1/info`** | **Server capabilities (NEW)** | — |
| **`POST`** | **`/api/v1/bench/run_one`** | **Submit benchmark case (NEW)** | — |
| **`GET`** | **`/api/v1/assets`** | **List local assets (NEW)** | — |
| `GET` | `/health` | Healthcheck + job counts | no |
| `GET` | `/` | Root info | no |
| `GET` | `/docs` | Swagger UI | no |

---

## C. New Endpoint Details

### `GET /api/v1/info`

Stable handshake for any external client (benchmark runner, mobile app, CI).

**Implementation:** `ai-core/src/api/routes/bench.py` → `get_info()`

**Response example:**

```json
{
  "api_version": "2.0.0",
  "results_contract_version": "2.0.0",
  "supports_video_source": ["upload", "local_asset"],
  "available_backends": ["cutie", "yolo"],
  "current_default_backend": "cutie",
  "active_optional_models": {
    "person_detection": false,
    "pose_estimation": false
  },
  "limits": {
    "max_upload_mb": 100,
    "supported_formats": [".avi", ".mkv", ".mov", ".mp4", ".webm"],
    "max_duration_sec": null,
    "max_concurrent_jobs": 1
  },
  "assets": {
    "root": "/Users/nicolas/Documents/app-mvp/data/bench_assets",
    "available": true
  },
  "server_build": {
    "git_sha": "8eabfc0",
    "build_time": null,
    "hostname": "MacBook-Air-de-Nicolas.local",
    "python_version": "3.11.9",
    "platform": "Darwin"
  },
  "timestamp": "2026-02-13T12:00:00.000000"
}
```

**Stability contract:** Fields will not be removed without a major version bump. Additional fields may appear in minor versions.

---

### `POST /api/v1/bench/run_one`

Submit one video case for analysis. Returns the same `job_id` used by all standard polling endpoints — no new polling logic needed.

**Implementation:** `ai-core/src/api/routes/bench.py` → `bench_run_one()`

**Content-Type:** `multipart/form-data`

#### Form fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video_source_type` | string | yes | `"upload"` or `"local_asset"` |
| `file` | file | if upload | Video file (mp4/mov/avi/mkv/webm, max 100 MB) |
| `asset_id` | string | if local_asset | Filename stem in `DATASETS_ROOT` |
| `asset_path` | string | alt for local_asset | Path (validated inside `DATASETS_ROOT`) |
| `disc_center_x` | float | no | Disc center X in pixels (seed) |
| `disc_center_y` | float | no | Disc center Y in pixels (seed) |
| `disc_radius` | float | no | Disc radius in pixels (seed) |
| `seed_frame` | int | no | Frame index for seed (default 0) |
| `case_id` | string | no | Stable test-case ID in runner's dataset |
| `client_run_id` | string | no | Global run ID on the client side |
| `tags` | string | no | JSON-encoded dict `{"key": "value"}` |
| `backend` | string | no | Override backend for this job (`"cutie"` or `"yolo"`) |

#### Canonical seed contract (no aliases)

For both `video_source_type=upload` and `video_source_type=local_asset`, the seed fields are:

- `disc_center_x`
- `disc_center_y`
- `disc_radius`
- `seed_frame` (optional, default `0`)

These are the only accepted names in the current server contract for benchmark runs.

**Not supported in `/api/v1/bench/run_one`:**
- `cx`, `cy`, `r`
- Nested payload variants like `seed.circle.cx` / `seed.circle.cy` / `seed.circle.r`

If any unsupported alias is sent, the server does not map it to disc selection and Cutie can fail with missing seed.

#### Response (201 / 200)

```json
{
  "job_id": "abc12345-def6",
  "case_id": "snatch_001",
  "client_run_id": "run_2026_02_13",
  "source_type": "upload",
  "status": "pending",
  "message": "Benchmark case queued. source=upload, case_id=snatch_001",
  "links": {
    "status":  "/api/v1/videos/abc12345-def6/status",
    "results": "/api/v1/videos/abc12345-def6/results",
    "delete":  "/api/v1/videos/abc12345-def6"
  }
}
```

#### Error codes

| Code | Reason |
|------|--------|
| 400 | Invalid file extension |
| 404 | `asset_id` not found in `DATASETS_ROOT` |
| 413 | File exceeds 100 MB |
| 422 | Missing required field or invalid `tags` JSON |
| 503 | `DATASETS_ROOT` not configured/does not exist |

---

### `GET /api/v1/assets`

List video files available in `DATASETS_ROOT` as local assets.

**Implementation:** `ai-core/src/api/routes/bench.py` → `list_assets()`

**Response example:**

```json
{
  "assets_root": "/Users/nicolas/Documents/app-mvp/data/bench_assets",
  "root_exists": true,
  "count": 2,
  "assets": [
    { "asset_id": "video_test_1", "filename": "video_test_1.mp4", "size_mb": 12.4 },
    { "asset_id": "snatch_001",   "filename": "snatch_001.mp4",   "size_mb": 8.7  }
  ]
}
```

> Note: `path` is intentionally omitted from the response for security.

---

## D. Assets (local_asset) Configuration

### Environment variable

```bash
export DATASETS_ROOT=/path/to/your/bench/videos
```

**Default (if not set):** `<project_root>/data/bench_assets`

### How asset_id resolution works

1. `asset_id` is treated as the **filename stem** (without extension).
2. The server tries each supported extension in order: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.
3. The resolved absolute path is validated to be **inside `DATASETS_ROOT`** (path-traversal prevention).
4. `asset_path` is an alias that accepts a relative or absolute path, also validated inside `DATASETS_ROOT`.

### Setup for local testing

```bash
mkdir -p data/bench_assets
cp your_video.mp4 data/bench_assets/snatch_001.mp4
# Optional: set env var to a different root
export DATASETS_ROOT=$(pwd)/data/bench_assets
```

---

## E. Job Persistence — Artefact Layout

Every job (benchmark or normal) produces:

```
data/api/results/<job_id>/
├── result.json          — ResultsContract v2.0.0 (disc tracks + metrics + summary)
├── job_meta.json        — NEW: full trazabilidad (see schema below)
├── pipeline.log         — FIXED: per-job log with step-by-step trace
├── pipeline_config.yaml — resolved YAML fed to PipelineRunner
└── metrics_calculator_output.json  — raw metrics from pipeline step
```

### `job_meta.json` schema

```json
{
  "job_id": "abc12345-def6",
  "schema_version": "1.0",
  "timestamps": {
    "created_at":  "2026-02-13T12:00:00",
    "started_at":  "2026-02-13T12:00:05",
    "finished_at": "2026-02-13T12:02:30",
    "duration_s":  145.2
  },
  "source": {
    "type": "upload",
    "original_filename": "snatch_001.mp4",
    "asset_id": null
  },
  "benchmark": {
    "case_id": "snatch_001",
    "client_run_id": "run_2026_02_13",
    "tags": { "env": "indoor", "disc": "black" }
  },
  "model_config": {
    "tracking_backend": "cutie",
    "enable_person_detection": false,
    "enable_pose_estimation": false
  },
  "video_metadata": {
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "duration_s": 3.2,
    "total_frames": 96
  },
  "status": "completed",
  "results_contract_version": "2.0.0"
}
```

Fields are `null` when not available (e.g. `asset_id` on upload jobs, `build_time` not tracked).

### `pipeline.log`

Fixed in this release: previously `logging.basicConfig` was called once and subsequent runs produced empty log files. Now `PipelineRunner._setup_logging` creates a **unique logger per run** (`PipelineRunner.<run_id>`) and adds handlers directly, bypassing `basicConfig`.

---

## F. Job Execution Model

- **Queue:** `asyncio.Queue` — single background worker (one job at a time).
- **Thread isolation:** the pipeline runs in `asyncio.to_thread` (blocking, safe).
- **Worker failure isolation:** errors in one job are caught, stored as `status=failed`, and the worker continues with the next job.
- **Progress:** 15 % → 85 % linear across pipeline steps via `on_step_start` callback.

### Processing status lifecycle

```
PENDING → PROCESSING → COMPLETED
                     ↘ FAILED
```

`FAILED` always includes a `reason` string in `job.message` and the full traceback in `job.error`. A `job_meta.json` with `"status": "failed"` is also written.

---

## G. Quickstart with curl

### 1. Check capabilities

```bash
curl -s http://localhost:8000/api/v1/info | python3 -m json.tool
```

### 2. Run one case — upload mode

```bash
curl -s -X POST http://localhost:8000/api/v1/bench/run_one \
  -F "video_source_type=upload" \
  -F "file=@/path/to/snatch_001.mp4" \
  -F "disc_center_x=640" \
  -F "disc_center_y=400" \
  -F "disc_radius=45" \
  -F "seed_frame=0" \
  -F "case_id=snatch_001" \
  -F "client_run_id=run_20260213" \
  -F 'tags={"env":"indoor","disc":"black"}' \
  | python3 -m json.tool
```

Save the returned `job_id`, e.g. `JOB=abc12345-def6`.

### 3. Run one case — local_asset mode

```bash
# Place asset first:
# cp snatch_001.mp4 data/bench_assets/snatch_001.mp4

curl -s -X POST http://localhost:8000/api/v1/bench/run_one \
  -F "video_source_type=local_asset" \
  -F "asset_id=snatch_001" \
  -F "disc_center_x=640" \
  -F "disc_center_y=400" \
  -F "disc_radius=45" \
  -F "seed_frame=0" \
  -F "case_id=snatch_001" \
  -F "client_run_id=run_20260213" \
  | python3 -m json.tool
```

### 4. Poll status

```bash
curl -s http://localhost:8000/api/v1/videos/$JOB/status | python3 -m json.tool
```

Typical response while processing:
```json
{
  "video_id": "abc12345-def6",
  "status": "processing",
  "progress": 0.45,
  "current_step": "cutie_disc_tracking",
  "message": "Tracking disc (3/5)",
  "created_at": "...",
  "updated_at": "..."
}
```

### 5. Get results

```bash
curl -s http://localhost:8000/api/v1/videos/$JOB/results | python3 -m json.tool
```

### 6. List available local assets

```bash
curl -s http://localhost:8000/api/v1/assets | python3 -m json.tool
```

### 7. Verify artefacts on disk

```bash
ls data/api/results/$JOB/
# result.json  job_meta.json  pipeline.log  pipeline_config.yaml
cat data/api/results/$JOB/job_meta.json | python3 -m json.tool
```

---

## H. Benchmark Runner Design (External Tool)

The runner is **not part of this server**. It lives in a separate script/repo and follows this loop:

```python
for case in dataset:
    resp = requests.post(
        f"{SERVER}/api/v1/bench/run_one",
        data={
            "video_source_type": "local_asset",
            "asset_id": case.asset_id,
            "case_id": case.case_id,
            "client_run_id": RUN_ID,
            "tags": json.dumps(case.tags),
            "disc_center_x": case.seed.cx,
            "disc_center_y": case.seed.cy,
            "disc_radius": case.seed.r,
        },
    )
    job_id = resp.json()["job_id"]

    # poll until done
    while True:
        status = requests.get(f"{SERVER}/api/v1/videos/{job_id}/status").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(3)

    results = requests.get(f"{SERVER}/api/v1/videos/{job_id}/results").json()
    # compare results vs ground truth, store metrics...
```

Key properties:
- **One video per call** — no batch endpoint needed.
- **Independent jobs** — a failure on case N does not affect case N+1.
- **Trazabilidad** — `case_id` + `client_run_id` appear in `job_meta.json` for offline analysis.

---

## I. Future Work (Not Yet Implemented)

| Item | Notes |
|------|-------|
| Raw vs post-processed results | Return both pre- and post-smoothing data for validation |
| Ground-truth comparison endpoint | `POST /api/v1/bench/compare` — server-side GT diff |
| Batch submission | `POST /api/v1/bench/run_batch` — multiple cases, returns list of job IDs |
| Runtime metrics in job_meta | CPU/GPU usage, memory peak |
| `build_time` in server_build | Inject at build/deploy time via env var |
| Capabilities versioning | Semver for `results_contract_version` bumps |
| Auth | API key header for multi-user deployments |

---

## J. Compatibility with Mobile App

**No breaking changes** — all `/api/v1/videos/*` endpoints are unchanged.

New endpoints (`/info`, `/bench/run_one`, `/assets`) are additive. The mobile app can optionally call `/api/v1/info` to verify server compatibility before uploading.

`job_meta.json` is a server-side artefact and is not returned to the mobile app as part of any existing response. The mobile app continues to use `/videos/{id}/results` for all result data.

---

*Last updated: February 2026 — v1.0 (benchmark endpoints implemented)*
