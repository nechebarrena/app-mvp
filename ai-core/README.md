# AI Core

Python processing engine for video analysis using YOLO models.

## Structure

```
ai-core/
├── configs/              # Pipeline configurations
├── models/
│   ├── custom/           # Trained models (best.pt)
│   └── pretrained/       # COCO models
├── src/
│   ├── domain/           # Entities, ports, label_mapper
│   ├── input_layer/      # Video loading
│   ├── perception/       # YOLO detectors
│   ├── analysis/         # Tracking, fusion
│   ├── visualization/    # Video renderers
│   └── pipeline/         # Orchestration
└── run_pipeline.py       # Entry point
```

## Usage

```bash
PYTHONPATH=src:. uv run python run_pipeline.py configs/compare_models.yaml
```

## Key Modules

| Module | Description |
|--------|-------------|
| `yolo_detector` | Generic YOLO (detect/segment/pose) |
| `multi_model_renderer` | Multi-panel comparison video |
| `label_mapper` | Unify labels across models |
