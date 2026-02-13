# AI Core

Python processing engine for video analysis using YOLO and Cutie VOS models.

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
└── run_pipeline.py       # Entry point
```

## Usage

### YOLO-only pipeline (original)
```bash
PYTHONPATH=src:. uv run python run_pipeline.py configs/full_analysis.yaml
```

### Cutie + YOLO hybrid pipeline (recommended for disc tracking)
```bash
PYTHONPATH=vendors/cutie:src:. uv run python run_pipeline.py configs/cutie_analysis.yaml
```

### Validation tool (test Cutie on a specific video)
```bash
PYTHONPATH=vendors/cutie:src:. uv run python ../tools/test_cutie.py <video> <selection.json> --save-video
```

## Key Modules

| Module | Description |
|--------|-------------|
| `yolo_detector` | Generic YOLO (detect/segment/pose) |
| `cutie_tracker` | Cutie VOS semi-supervised tracker |
| `detection_merger` | Merge detections from multiple models |
| `model_tracker` | Kalman + Hungarian object tracker |
| `metrics_calculator` | Physics metrics (velocity, energy, power) |
| `multi_model_renderer` | Multi-panel comparison video |

## Tracking Backends

| Backend | Disc Detection | Coverage | Requires Selection | Mobile-Ready |
|---------|---------------|----------|-------------------|-------------|
| YOLO | COCO "frisbee" class | ~10-50% | Optional | Yes (lightweight) |
| Cutie | Visual appearance | ~100% | Required | Future (with optimization) |

The recommended approach for server-side processing is the Cutie hybrid pipeline.
YOLO is maintained for person detection, pose estimation, and future mobile deployment.

## Setup

```bash
# Install dependencies
uv sync

# Download Cutie weights (required for Cutie pipeline)
curl -L -o models/pretrained/cutie/cutie-base-mega.pth \
  https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth
```
