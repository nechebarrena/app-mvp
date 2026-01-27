# App MVP: AI Video Analysis

A modular computer vision pipeline for video analysis using multiple YOLO models (detection, segmentation, pose estimation).

## Features

- **Multi-Model Pipeline**: Run 3+ YOLO models in parallel on the same video
- **Multi-Panel Visualization**: Compare model outputs side-by-side
- **Label Unification**: Map different model labels to unified concepts
- **Configurable**: YAML-based pipeline configuration

## Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (dependency management)

### Installation

```bash
cd ai-core
uv sync
```

### Run Multi-Model Comparison

```bash
cd ai-core
PYTHONPATH=src:. uv run python run_pipeline.py configs/compare_models.yaml
```

This runs:
- **yolo_custom**: Your trained segmentation model (atleta, barra, discos)
- **yolo_coco**: COCO pretrained segmentation (80 classes)
- **yolo_pose**: Pose estimation (person keypoints)

Output: `data/outputs/compare_models_run/comparison.mp4`

## Project Structure

```
/app-mvp
├── /ai-core                    # Python AI core
│   ├── /configs                # Pipeline YAML configs
│   ├── /models
│   │   ├── /custom             # Your trained models (best.pt)
│   │   └── /pretrained         # COCO models (yolov8s-seg.pt, etc.)
│   ├── /src                    # Source code
│   └── run_pipeline.py         # Entry point
├── /data
│   ├── /raw                    # Input videos
│   └── /outputs                # Pipeline outputs
└── /docs                       # Documentation
```

## Documentation

- **[Full Documentation](docs/README_FULL.md)**: Complete reference for all features
- **[Architecture](docs/architecture.md)**: Project structure and design
- **[Pipeline Guide](docs/pipeline_guide.md)**: Configuration and execution

## Available Models

| Model | Type | Classes |
|-------|------|---------|
| models/custom/best.pt | Segment | atleta, barra, discos |
| models/pretrained/yolov8s-seg.pt | Segment | 80 COCO classes |
| models/pretrained/yolov8n-pose.pt | Pose | person (17 keypoints) |

## Label Mapping

Unify different model labels:

```yaml
label_mapping:
  atleta:
    yolo_custom: "atleta"
    yolo_coco: "person"
    yolo_pose: "person"
  disco:
    yolo_custom: "discos"
    yolo_coco: "frisbee"
```
