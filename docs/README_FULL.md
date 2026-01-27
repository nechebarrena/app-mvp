# APP-MVP: AI Video Analysis - Complete Documentation

> **Generated:** January 2026  
> **Purpose:** Complete reference document for AI review and project understanding

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Pipeline System](#4-pipeline-system)
5. [Available Modules](#5-available-modules)
6. [Model Management](#6-model-management)
7. [Label Mapping System](#7-label-mapping-system)
8. [Configuration Examples](#8-configuration-examples)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Current Status & Capabilities](#10-current-status--capabilities)

---

## 1. Project Overview

### Goal
Build an MVP for mobile video analysis that:
1. Captures short video clips (10-60s)
2. Processes them using AI (Object Detection + Segmentation + Pose Estimation)
3. Calculates derived metrics (trajectory, counts, etc.)
4. Displays results to the user

### Architecture Philosophy
- **Python AI Core**: R&D lab and processing engine
- **Modular Pipeline**: YAML-configured, plug-and-play modules
- **Multi-Model Support**: Run multiple YOLO models in parallel
- **Label Unification**: Map different model outputs to unified concepts

---

## 2. Architecture

### Design Principles

1. **Domain-Driven Design (DDD)**
   - Entities: Data structures (VideoSession, Detection, TrackedObject)
   - Ports: Interfaces (IPipelineStep)
   - Decoupled modules

2. **Hybrid Pipeline**
   - Memory mode: Direct Python object passing (production)
   - Disk mode: JSON serialization (debugging)

3. **Multi-Model Fusion**
   - Run detection, segmentation, and pose models in parallel
   - Label mapping for concept unification across models

---

## 3. Directory Structure

```
/app-mvp
├── /ai-core                    # PYTHON CORE
│   ├── /configs                # Pipeline YAML configurations
│   │   ├── compare_models.yaml # Multi-model comparison (main config)
│   │   ├── lifting_test.yaml   # Full lifting analysis pipeline
│   │   └── ...
│   ├── /models                 # Model weights
│   │   ├── /custom             # Custom trained models
│   │   │   └── best.pt         # Weightlifting segmentation (atleta, barra, discos)
│   │   └── /pretrained         # Standard pretrained models
│   │       ├── yolov8s-seg.pt  # COCO segmentation (80 classes)
│   │       ├── yolov8s.pt      # COCO detection (80 classes)
│   │       ├── yolov8n-pose.pt # COCO pose estimation
│   │       └── yolov8n-seg.pt  # COCO segmentation nano
│   ├── /src
│   │   ├── /domain             # Core contracts
│   │   │   ├── entities.py     # Detection, TrackedObject, VideoSession
│   │   │   ├── ports.py        # IPipelineStep interface
│   │   │   └── label_mapper.py # Label unification system
│   │   ├── /input_layer        # Data ingestion
│   │   │   └── video_loader.py
│   │   ├── /perception         # AI models
│   │   │   ├── yolo_detector.py    # Generic YOLO (detect/segment/pose)
│   │   │   ├── yolo_segmentor.py   # Legacy segmentation
│   │   │   └── yolo_pose.py        # Legacy pose
│   │   ├── /analysis           # Business logic
│   │   │   ├── merger.py           # Detection merger
│   │   │   ├── tracking/           # ByteTrack
│   │   │   ├── disc_calibrator.py  # Disc prior calibration
│   │   │   ├── disc_fusion_tracker.py # Kalman fusion
│   │   │   └── lifting_optimizer.py
│   │   ├── /visualization      # Output renderers
│   │   │   ├── multi_model_renderer.py # Multi-panel comparison
│   │   │   └── video_renderer.py       # Single overlay
│   │   └── /pipeline           # Orchestration
│   │       ├── runner.py
│   │       └── config.py
│   ├── run_pipeline.py         # Main entry point
│   └── pyproject.toml          # Dependencies (uv)
│
├── /data                       # DATA STORAGE (gitignored)
│   ├── /raw                    # Input videos
│   ├── /outputs                # Pipeline outputs per run
│   └── /processed              # Intermediate files
│
├── /docs                       # Documentation
│   ├── architecture.md
│   ├── pipeline_guide.md
│   ├── pipeline_logic.md
│   ├── lifting_heuristics.md
│   └── README_FULL.md          # This file
│
└── /mobile-app                 # Future mobile client
```

---

## 4. Pipeline System

### Configuration Schema

```yaml
session:
  video_id: "video_test_1"      # Video filename (without extension)
  output_dir: "my_run"          # Output folder name

steps:
  - name: step_name             # Unique identifier
    module: "registered_module" # Module from registry
    enabled: true               # Can disable steps
    input_source: "memory"      # "memory" or "disk"
    input_from_step: "previous" # Explicit input source (or list)
    save_output: true           # Save to JSON
    params:                     # Module-specific parameters
      key: value
```

### Execution Flow

1. **Linear**: Steps run sequentially, each consuming previous output
2. **Branching**: Multiple steps can consume the same source (parallel)
3. **Merging**: A step can consume outputs from multiple steps

```
Ingestion ─┬─> YOLO Custom ──┬─> Merger ─> Visualization
           ├─> YOLO COCO ────┤
           └─> YOLO Pose ────┘
```

---

## 5. Available Modules

### Input Layer
| Module | Class | Description |
|--------|-------|-------------|
| `video_loader` | VideoLoader | Loads video, creates VideoSession |

### Perception
| Module | Class | Description |
|--------|-------|-------------|
| `yolo_detector` | YoloDetector | **Generic YOLO** (detect/segment/pose) |
| `yolo_segmentor` | YoloSegmentor | Legacy segmentation |
| `yolo_pose` | YoloPoseDetector | Legacy pose |

### Analysis
| Module | Class | Description |
|--------|-------|-------------|
| `detection_merger` | DetectionMerger | Combines detections from branches |
| `byte_tracker` | ByteTracker | Associates detections across frames |
| `disc_calibrator` | DiscConsensusCalibrator | Builds disc prior from consensus |
| `disc_fusion_tracker` | DiscFusionTracker | Kalman + multi-source fusion |
| `trajectory_cleaner` | TrajectoryCleaner | Filters noisy tracks |
| `lifting_optimizer` | LiftingSessionOptimizer | Weightlifting-specific logic |

### Visualization
| Module | Class | Description |
|--------|-------|-------------|
| `multi_model_renderer` | MultiModelRenderer | **Multi-panel comparison video** |
| `video_renderer` | VideoOverlayRenderer | Single overlay video |

---

## 6. Model Management

### Model Directory Structure

```
ai-core/models/
├── custom/
│   └── best.pt              # Custom trained (52MB)
│                            # Classes: atleta, barra, discos
└── pretrained/
    ├── yolov8s-seg.pt       # COCO segmentation small (24MB)
    ├── yolov8s.pt           # COCO detection small (23MB)
    ├── yolov8n-pose.pt      # COCO pose nano (7MB)
    └── yolov8n-seg.pt       # COCO segmentation nano (7MB)
```

### Available Models

| Model | Type | Classes | Size | Speed |
|-------|------|---------|------|-------|
| best.pt | Segment | atleta, barra, discos | 52MB | ~3 fps |
| yolov8s-seg.pt | Segment | 80 COCO classes | 24MB | ~5 fps |
| yolov8n-pose.pt | Pose | person (17 keypoints) | 7MB | ~13 fps |

---

## 7. Label Mapping System

### Purpose
Unify different label names across models into a single concept space.

### Configuration

```yaml
# In visualization params
label_mapping:
  atleta:                    # Global label
    yolo_custom: "atleta"    # Model-specific mapping
    yolo_coco: "person"
    yolo_pose: "person"
  disco:
    yolo_custom: "discos"
    yolo_coco: "frisbee"
  barra:
    yolo_custom: "barra"
    # No COCO equivalent

visualize_labels: ["atleta", "disco", "barra"]  # Filter
use_global_labels: true  # Display unified names
```

### Features
- Maps model-specific labels to global concepts
- Filters output to show only relevant labels
- Ensures consistent colors across panels
- Handles missing mappings gracefully

---

## 8. Configuration Examples

### Multi-Model Comparison (Current Main Pipeline)

```yaml
# configs/compare_models.yaml
session:
  video_id: "video_test_1"
  output_dir: "compare_models_run"

steps:
  - name: ingestion
    module: "video_loader"
    enabled: true
    input_source: "disk"

  - name: yolo_custom
    module: "yolo_detector"
    enabled: true
    input_from_step: "ingestion"
    params:
      model_path: "models/custom/best.pt"
      task: "segment"
      source_name: "yolo_custom"

  - name: yolo_coco
    module: "yolo_detector"
    enabled: true
    input_from_step: "ingestion"
    params:
      model_path: "models/pretrained/yolov8s-seg.pt"
      task: "segment"
      source_name: "yolo_coco"

  - name: yolo_pose
    module: "yolo_detector"
    enabled: true
    input_from_step: "ingestion"
    params:
      model_path: "models/pretrained/yolov8n-pose.pt"
      task: "pose"
      source_name: "yolo_pose"

  - name: comparison_video
    module: "multi_model_renderer"
    enabled: true
    input_from_step: ["yolo_custom", "yolo_coco", "yolo_pose"]
    params:
      video_source: "../data/raw/video_test_1.mp4"
      output_filename: "../data/outputs/compare_models_run/comparison.mp4"
      panel_width: 640
      mask_alpha: 0.45
      mask_contour: true
      label_mapping:
        atleta:
          yolo_custom: "atleta"
          yolo_coco: "person"
          yolo_pose: "person"
        disco:
          yolo_custom: "discos"
          yolo_coco: "frisbee"
        barra:
          yolo_custom: "barra"
      visualize_labels: ["atleta", "disco", "barra"]
```

---

## 9. Running the Pipeline

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
cd ai-core
uv sync
```

### Execution

```bash
cd ai-core
PYTHONPATH=src:. uv run python run_pipeline.py configs/compare_models.yaml
```

### Output Structure

```
data/outputs/compare_models_run/
├── pipeline.log                    # Execution log
├── yolo_custom_output.json         # Detections from custom model
├── yolo_coco_output.json           # Detections from COCO model
├── yolo_pose_output.json           # Detections from pose model
├── comparison.mp4                  # Multi-panel video
└── comparison_video_output.json    # Video path
```

---

## 10. Current Status & Capabilities

### ✅ Implemented & Working

1. **Multi-Model Pipeline**
   - Run 3+ YOLO models in parallel
   - Generic YoloDetector supports detect/segment/pose
   - Progress indicators with ETA

2. **Multi-Panel Visualization**
   - One panel per model
   - Segmentation masks with configurable opacity
   - Contour outlines for visibility
   - Frame counter in each panel
   - Class legend from frame 0

3. **Label Mapping System**
   - Unify labels across models
   - Filter by global labels
   - Consistent colors for same concept

4. **Model Organization**
   - Clean folder structure (custom/pretrained)
   - All configs use correct paths

### 🔄 Available but Not Primary Focus

1. **Tracking Pipeline** (ByteTrack, Kalman fusion)
2. **Disc Calibration** (consensus-based)
3. **Lifting Optimizer** (domain-specific heuristics)

### 📋 Future Work

1. **Detection Merger Enhancements**
   - Spatial deduplication when models detect same object
   - Confidence fusion strategies

2. **Temporal Consistency**
   - Track consistency across frames
   - Gap filling for missed detections

3. **Mobile Export**
   - CoreML / TFLite conversion
   - Optimized inference

---

## Appendix: Detection Entity Schema

```python
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    mask: Optional[List[List[float]]]        # Polygon points
    keypoints: Optional[List[List[float]]]   # [[x, y, conf], ...]
    source: Optional[str]                    # "yolo_custom", "yolo_coco", etc.
    radius_px: Optional[float]               # For circle detections
    shape_score: Optional[float]             # Quality metric
    debug: Optional[Dict[str, Any]]          # Extra info
```

---

*End of Documentation*
