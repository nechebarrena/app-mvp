# APP-MVP: AI Video Analysis - Complete Documentation

> **Generated:** January 2026  
> **Purpose:** Complete reference document for AI review and project understanding

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Three-Phase Heuristics Architecture](#4-three-phase-heuristics-architecture)
5. [Pipeline System](#5-pipeline-system)
6. [Available Modules](#6-available-modules)
7. [Model Management](#7-model-management)
8. [Label Mapping System](#8-label-mapping-system)
9. [Configuration Examples](#9-configuration-examples)
10. [Running the Pipeline](#10-running-the-pipeline)
11. [Current Status & Capabilities](#11-current-status--capabilities)

---

## 1. Project Overview

### Goal
Build an MVP for mobile video analysis that:
1. Captures short video clips (10-60s) of weightlifting sessions
2. Processes them using AI (Object Detection + Segmentation + Pose Estimation)
3. Tracks objects of interest (disc, athlete, barbell)
4. Applies domain heuristics (single athlete, single disc, size constraints)
5. Calculates derived metrics (trajectory, counts, etc.)
6. Displays results to the user

### Architecture Philosophy
- **Python AI Core**: R&D lab and processing engine
- **Modular Pipeline**: YAML-configured, plug-and-play modules
- **Multi-Model Support**: Run multiple YOLO models in parallel
- **Three-Phase Heuristics**: Pre-tracking → Tracking → Post-tracking separation
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

4. **Three-Phase Heuristics** (NEW)
   - Pre-tracking: Stateless filters (size, confidence, largest-selector)
   - Tracking: Temporal association (Kalman, Hungarian)
   - Post-tracking: Trajectory refinement (smoothing, outlier removal)

---

## 3. Directory Structure

```
/app-mvp
├── /ai-core                    # PYTHON CORE
│   ├── /configs                # Pipeline YAML configurations
│   │   ├── single_disc_tracking.yaml # Main 3-phase tracking pipeline
│   │   ├── compare_models.yaml       # Multi-model comparison
│   │   ├── tracking_comparison.yaml  # Tracking comparison
│   │   └── ...
│   ├── /models                 # Model weights
│   │   ├── /custom             # Custom trained models
│   │   │   └── best.pt         # Weightlifting segmentation (atleta, barra, discos)
│   │   └── /pretrained         # Standard pretrained models
│   │       ├── yolov8s-seg.pt  # COCO segmentation (80 classes)
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
│   │   │   ├── detection_filter.py   # PRE-TRACKING heuristics
│   │   │   ├── model_tracker.py      # TRACKING (Kalman + Hungarian)
│   │   │   ├── track_refiner.py      # POST-TRACKING refinement
│   │   │   ├── merger.py             # Detection merger
│   │   │   └── lifting_optimizer.py  # Domain-specific logic
│   │   ├── /visualization      # Output renderers
│   │   │   ├── multi_model_renderer.py # Multi-panel comparison
│   │   │   └── video_renderer.py       # Single overlay
│   │   ├── /tools              # Utility tools
│   │   │   ├── disc_selector.py      # GUI for manual disc selection
│   │   │   └── selection_loader.py   # Load selection into pipeline
│   │   └── /pipeline           # Orchestration
│   │       ├── runner.py
│   │       └── config.py
│   ├── run_pipeline.py         # Main entry point
│   ├── select_disc.py          # Manual disc selection tool
│   └── pyproject.toml          # Dependencies (uv)
│
├── /data                       # DATA STORAGE (gitignored)
│   ├── /raw                    # Input videos
│   ├── /outputs                # Pipeline outputs per run
│   │   ├── disc_selection.json # Manual disc selection
│   │   └── single_disc_3phase_run/
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

## 4. Three-Phase Heuristics Architecture

The pipeline separates heuristics into three phases based on what information they need:

### Phase 1: Pre-Tracking (DetectionFilter)

**Responsibility**: Reduce candidates using STATELESS filters (no temporal information).

```
Detection (YOLO) → [DetectionFilter] → Filtered Detections
```

**Available Filters**:
| Filter | Description | Example |
|--------|-------------|---------|
| `min_confidence` | Minimum detection score | Reject score < 0.05 |
| `size_filter` | Filter by expected size | Disc must be ~125px ±30% |
| `roi_filter` | Static region of interest | Only detections in bottom half |
| `initial_selector` | Best match in frame 0 | Closest disc to manual selection |
| `largest_selector` | Keep only largest per class | One athlete (the largest) per frame |

**Example Config**:
```yaml
params:
  min_confidence: 0.05
  size_filter:
    enabled: true
    selection_file: "../data/outputs/disc_selection.json"
    tolerance: 0.30
    classes: ["discos"]
  largest_selector:
    enabled: true
    classes: ["atleta"]  # Only one athlete per frame
```

### Phase 2: Tracking (ModelTracker)

**Responsibility**: Temporal association using Kalman filter and Hungarian algorithm.

```
Filtered Detections → [ModelTracker] → Tracked Objects
```

**Features**:
- **Kalman Filter**: 2D position + velocity model for prediction
- **Hungarian Assignment**: Optimal matching between tracks and detections
- **Track Lifecycle**: TENTATIVE → CONFIRMED → LOST → TERMINATED
- **Single-Object Mode**: For classes where only one object should be tracked
- **Dual Threshold**: High-confidence for new tracks, low-confidence for maintaining tracks

**Example Config**:
```yaml
params:
  enabled: true
  classes_to_track: ["discos", "atleta", "barra"]
  initial_selection:
    class_name: "discos"
    selection_file: "../data/outputs/disc_selection.json"
  min_det_score: 0.05
  high_det_score: 0.15
  max_age_frames: 30
  association:
    max_center_dist_px: 200
```

### Phase 3: Post-Tracking (TrackRefiner)

**Responsibility**: Refine trajectories using COMPLETE track information.

```
Tracked Objects → [TrackRefiner] → Refined Tracked Objects
```

**Available Refinements**:
| Refinement | Description |
|------------|-------------|
| `smoothing` | Moving average, exponential, or Savitzky-Golay filter |
| `outlier_removal` | Remove sudden jumps based on velocity statistics |
| `direction_constraints` | (Future) Validate trajectory direction |

**Example Config**:
```yaml
params:
  enabled: true
  classes_to_refine: ["discos"]
  smoothing:
    enabled: true
    method: "moving_average"
    window: 5
  outlier_removal:
    enabled: false
    threshold_std: 3.0
```

### Why This Separation?

| Aspect | Pre-Tracking | Tracking | Post-Tracking |
|--------|--------------|----------|---------------|
| **State** | None | Temporal (Kalman) | Full trajectory |
| **Purpose** | Reduce noise | Assign IDs | Refine output |
| **A/B Testing** | Easy | Easy | Easy |
| **Modularity** | Independent | Independent | Independent |

---

## 5. Pipeline System

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

```
Ingestion ─┬─> YOLO Custom ─> Filter ─> Track ─> Refine ─┬─> Visualization
           ├─> YOLO COCO ──> Filter ─> Track ─> Refine ──┤
           └─> YOLO Pose ─────────────────────────────────┘
```

---

## 6. Available Modules

### Input Layer
| Module | Class | Description |
|--------|-------|-------------|
| `video_loader` | VideoLoader | Loads video, creates VideoSession |
| `selection_loader` | SelectionLoader | Loads manual disc selection |

### Perception
| Module | Class | Description |
|--------|-------|-------------|
| `yolo_detector` | YoloDetector | **Generic YOLO** (detect/segment/pose) |
| `yolo_segmentor` | YoloSegmentor | Legacy segmentation |
| `yolo_pose` | YoloPoseDetector | Legacy pose |

### Analysis (Three-Phase)
| Module | Class | Phase | Description |
|--------|-------|-------|-------------|
| `detection_filter` | DetectionFilter | **Pre-Tracking** | Size, confidence, largest-selector |
| `model_tracker` | ModelTracker | **Tracking** | Kalman + Hungarian + single-object |
| `track_refiner` | TrackRefiner | **Post-Tracking** | Trajectory smoothing |
| `detection_merger` | DetectionMerger | - | Combines detections from branches |
| `lifting_optimizer` | LiftingSessionOptimizer | - | Domain-specific logic |

### Visualization
| Module | Class | Description |
|--------|-------|-------------|
| `multi_model_renderer` | MultiModelRenderer | **Multi-panel comparison video** |
| `video_renderer` | VideoOverlayRenderer | Single overlay video |

### Tools
| Tool | Description |
|------|-------------|
| `select_disc.py` | GUI for manual disc center/radius selection |

---

## 7. Model Management

### Model Directory Structure

```
ai-core/models/
├── custom/
│   └── best.pt              # Custom trained (52MB)
│                            # Classes: atleta, barra, discos
└── pretrained/
    ├── yolov8s-seg.pt       # COCO segmentation small (24MB)
    ├── yolov8n-pose.pt      # COCO pose nano (7MB)
    └── yolov8n-seg.pt       # COCO segmentation nano (7MB)
```

### Available Models

| Model | Type | Classes | Size | Speed |
|-------|------|---------|------|-------|
| best.pt | Segment | atleta, barra, discos | 52MB | ~3 fps |
| yolov8s-seg.pt | Segment | 80 COCO classes | 24MB | ~7 fps |
| yolov8n-pose.pt | Pose | person (17 keypoints) | 7MB | ~18 fps |

---

## 8. Label Mapping System

### Purpose
Unify different label names across models into a single concept space.

### Configuration

```yaml
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

visualize_labels: ["atleta", "disco", "barra"]  # Filter
use_global_labels: true  # Display unified names
```

---

## 9. Configuration Examples

### Main Pipeline: Single Disc Tracking (3-Phase)

```yaml
# configs/single_disc_tracking.yaml
session:
  video_id: "video_test_1"
  output_dir: "single_disc_3phase_run"

steps:
  # Stage 0: Video Loading
  - name: ingestion
    module: "video_loader"

  # Stage 1: Detection
  - name: yolo_custom_detection
    module: "yolo_detector"
    params:
      model_path: "models/custom/best.pt"
      task: "segment"
      source_name: "yolo_custom"

  - name: yolo_coco_detection
    module: "yolo_detector"
    params:
      model_path: "models/pretrained/yolov8s-seg.pt"
      task: "segment"
      source_name: "yolo_coco"

  # Stage 2: Pre-Tracking Filter
  - name: yolo_custom_filtered
    module: "detection_filter"
    input_from_step: "yolo_custom_detection"
    params:
      size_filter:
        enabled: true
        classes: ["discos"]
        tolerance: 0.30
      largest_selector:
        enabled: true
        classes: ["atleta"]

  # Stage 3: Tracking
  - name: yolo_custom_tracked
    module: "model_tracker"
    input_from_step: "yolo_custom_filtered"
    params:
      initial_selection:
        class_name: "discos"
        selection_file: "../data/outputs/disc_selection.json"
      max_age_frames: 30

  # Stage 4: Post-Tracking Refinement
  - name: yolo_custom_refined
    module: "track_refiner"
    input_from_step: "yolo_custom_tracked"
    params:
      smoothing:
        enabled: true
        method: "moving_average"
        window: 5

  # Stage 5: Visualization
  - name: tracking_video
    module: "multi_model_renderer"
    input_from_step: ["yolo_custom_refined", "yolo_coco_refined", "yolo_pose"]
```

---

## 10. Running the Pipeline

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
cd ai-core
uv sync
```

### Step 1: Manual Disc Selection

```bash
cd ai-core
PYTHONPATH=src:. uv run python select_disc.py
```

This opens a GUI to select the disc center and edge in frame 0. The selection is saved to `data/outputs/disc_selection.json`.

### Step 2: Run Pipeline

```bash
cd ai-core
PYTHONPATH=src:. uv run python run_pipeline.py configs/single_disc_tracking.yaml
```

### Output Structure

```
data/outputs/single_disc_3phase_run/
├── pipeline.log                           # Execution log
├── yolo_custom_detection_output.json      # Raw detections
├── yolo_custom_filtered_output.json       # After pre-filter
├── yolo_custom_tracked_output.json        # After tracking
├── yolo_custom_tracked_output_summary.json # Track statistics
├── yolo_custom_tracked_output_debug.jsonl  # Per-frame debug
├── yolo_custom_refined_output.json        # After refinement
├── tracked_comparison.mp4                 # Multi-panel video
└── ...
```

---

## 11. Current Status & Capabilities

### ✅ Implemented & Working

1. **Three-Phase Heuristics Architecture**
   - DetectionFilter: Size, confidence, largest-selector
   - ModelTracker: Kalman + Hungarian + single-object mode
   - TrackRefiner: Moving average smoothing

2. **Multi-Model Pipeline**
   - Run 3+ YOLO models in parallel
   - Generic YoloDetector supports detect/segment/pose
   - Progress indicators with ETA

3. **Multi-Panel Visualization**
   - One panel per model
   - Segmentation masks with configurable opacity
   - Track IDs and trajectories
   - Class legend from frame 0

4. **Manual Selection Tool**
   - GUI to select disc center and edge
   - Saves reference for tracking heuristics

5. **Domain Heuristics**
   - Single disc of interest (size constrained)
   - Single athlete of interest (largest in frame)

### 📋 Future Work

1. **Temporal Heuristics**
   - Trajectory direction constraints
   - Movement pattern validation

2. **Detection Merger**
   - Spatial deduplication when models detect same object
   - Confidence fusion strategies

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

class TrackedObject(BaseModel):
    track_id: int
    detection: Detection
    history: List[Tuple[float, float]]       # Center points
    velocity: Optional[Tuple[float, float]]
    smoothed_position: Optional[Tuple[float, float]]  # After refinement
```

---

*End of Documentation*
