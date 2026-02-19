# AI Core

Python processing engine for video analysis using Cutie VOS and YOLO models.

## Structure

```
ai-core/
├── configs/              # Pipeline configurations
│   ├── full_analysis.yaml      # YOLO-only pipeline
│   └── cutie_analysis.yaml     # Cutie + YOLO hybrid pipeline
├── models/
│   ├── custom/           # Trained models (best.pt)
│   └── pretrained/       # COCO models + Cutie weights
├── vendors/
│   └── cutie/            # Cutie VOS model (CVPR 2024)
├── src/
│   ├── domain/           # Entities, ports, label_mapper
│   ├── input_layer/      # Video loading
│   ├── perception/       # YOLO detectors + Cutie tracker
│   ├── analysis/         # Tracking, fusion, metrics
│   ├── visualization/    # Video renderers
│   ├── api/              # FastAPI backend
│   └── pipeline/         # Orchestration
├── control_panel.py      # Web-based control panel (Flask)
├── run_pipeline.py       # CLI entry point
└── run_api.py            # Direct FastAPI launch
```

## Quick Start

### Control Panel (recommended)
```bash
cd ai-core
PYTHONPATH=vendors/cutie:src:. uv run python control_panel.py
```
Opens a web dashboard at `http://localhost:5001` to manage services, configure models, process videos.

### CLI pipeline
```bash
PYTHONPATH=vendors/cutie:src:. uv run python run_pipeline.py configs/cutie_analysis.yaml
```

### Validation tool (test Cutie on a specific video)
```bash
PYTHONPATH=vendors/cutie:src:. uv run python ../tools/test_cutie.py <video> <selection.json> --save-video
```

## Model Configuration

The pipeline's model usage is configurable from the Control Panel:

| Model | Purpose | Default | Toggle |
|-------|---------|---------|--------|
| **Cutie VOS** | Disc tracking (mask + trajectory) | ON (required) | Backend selector |
| **YOLO person** | Athlete segmentation mask | OFF | Optional toggle |
| **YOLO pose** | Skeleton keypoints | OFF | Optional toggle |

**Disc-only mode (default):** Only Cutie runs. Fastest processing (~5 pipeline steps).

**With person/pose:** Adds YOLO steps. Slower but provides athlete overlay data.

## API Output Contract

The server always returns:
- `metadata` — video properties (fps, resolution, frames, duration)
- `tracks` — at least one disc track (`class_name="frisbee"`) with mask + trajectory
- `metrics` — 13 time series (position, velocity, energy, power)
- `summary` — peak values

See [`docs/api_guide.md`](../docs/api_guide.md) for the full specification.

## Key Modules

| Module | Description |
|--------|-------------|
| `cutie_tracker` | Cutie VOS semi-supervised tracker |
| `yolo_detector` | Generic YOLO (detect/segment/pose) |
| `detection_merger` | Merge detections from multiple models |
| `model_tracker` | Kalman + Hungarian object tracker |
| `metrics_calculator` | Physics metrics (velocity, energy, power) |

## Benchmark / Testing

Three extra endpoints support external benchmark tooling (no mobile impact):

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/info` | Capabilities handshake (versions, backends, limits) |
| `POST /api/v1/bench/run_one` | Submit one case — upload or local asset |
| `GET /api/v1/assets` | List videos in `DATASETS_ROOT` |

**Local asset setup:**
```bash
mkdir -p data/bench_assets
cp your_video.mp4 data/bench_assets/snatch_001.mp4
# or set a custom root:
export DATASETS_ROOT=/path/to/videos
```

See [`docs/BENCHMARK_API.md`](../docs/BENCHMARK_API.md) for full reference and curl examples.

## Per-job artefacts

Every processed job writes to `data/api/results/<job_id>/`:

```
result.json          — ResultsContract v2.0.0
job_meta.json        — timestamps, backend, case_id, tags, video metadata
pipeline.log         — full per-step trace
pipeline_config.yaml — resolved pipeline YAML
```

## Setup

```bash
# Install dependencies
uv sync

# Download Cutie weights (required)
curl -L -o models/pretrained/cutie/cutie-base-mega.pth \
  https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth
```
