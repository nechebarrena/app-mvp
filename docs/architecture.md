# Project Architecture

## Overview
This project is designed as an MVP for a mobile video analysis application focused on weightlifting. The core logic is developed in Python ("AI Core") with the intention of exporting optimized models to a native Mobile App.

The architecture strictly separates:
1.  **Data Storage** (`/data`): Video inputs and processing outputs.
2.  **Logic & Research** (`/ai-core`): The Python environment for developing models and algorithms.
3.  **Application** (`/mobile-app`): The future target deployment environment.

## 1. Directory Structure

```text
/app-mvp
│
├── /data                       <-- DATA STORAGE (Gitignored)
│   ├── /raw                    <-- Input videos + Sidecar JSON metadata
│   ├── /processed              <-- Intermediate debug frames
│   └── /outputs                <-- Final results (JSON reports, Overlay videos)
│
├── /ai-core                    <-- PYTHON CORE
│   ├── pyproject.toml          <-- Dependencies (managed by uv)
│   ├── /configs                <-- Pipeline YAML configurations
│   ├── /models                 <-- Model weights
│   │   ├── /custom             <-- Custom trained (best.pt)
│   │   └── /pretrained         <-- Standard YOLO models
│   ├── /vendors               <-- Vendored dependencies
│   │   └── /cutie             <-- Cutie VOS model (CVPR 2024)
│   └── /src                    <-- Production-ready Code
│       ├── /domain             <-- CONTRACTS (Entities & Interfaces)
│       ├── /input_layer        <-- Data Ingestion & ETL
│       ├── /perception         <-- Vision Models (YOLO, Cutie)
│       ├── /analysis           <-- Business Logic (Filter, Track, Refine)
│       ├── /visualization      <-- Debug Tools & Renderers
│       ├── /tools              <-- Utility Tools (disc_selector)
│       └── /pipeline           <-- Orchestration
│
├── /tools                      <-- Standalone validation & testing tools
│
└── /mobile-app                 <-- MOBILE CLIENT (Future)
```

## 2. Domain-Driven Design (DDD) & Contracts
To ensure modularity and testability, we use a simplified DDD approach located in `ai-core/src/domain`.

### Entities (`entities.py`)
Data structures that flow between modules. They are "dumb" data containers (Pydantic models or Dataclasses).
*   **VideoSession:** Represents a loaded video file and its metadata.
*   **FrameData:** A single frame image with timestamp.
*   **Detection:** Result of the object detector (bounding box, class, mask, keypoints).
*   **TrackedObject:** Result of the tracker (ID, history of positions, velocity, smoothed_position).

### Ports (`ports.py`)
Interfaces (Python Protocols) that define what a module *must do*, decoupling the *what* from the *how*.
*   **IPipelineStep:** Generic interface for any processing step, enforcing I/O capabilities.

### Label Mapper (`label_mapper.py`)
Utility class for mapping model-specific labels (e.g., "person", "frisbee") to global labels (e.g., "atleta", "disco").

## 3. Detection & Tracking Architecture

The pipeline supports two tracking backends for disc tracking:

### Backend A: YOLO + Three-Phase Heuristics (Original)

```
YOLO Detection → [Pre-Tracking] → [Tracking] → [Post-Tracking] → Output
                  DetectionFilter   ModelTracker   TrackRefiner
```

- **Phase 1 (DetectionFilter)**: Stateless per-frame filtering (size, confidence, ROI)
- **Phase 2 (ModelTracker)**: Temporal tracking (Kalman + Hungarian assignment)
- **Phase 3 (TrackRefiner)**: Trajectory smoothing

**Limitation**: YOLO uses COCO class labels ("frisbee", "sports ball") which are poor proxies for weightlifting discs. This causes sparse detections (often <10% frame coverage) and track fragmentation.

### Backend B: Cutie VOS (Recommended)

```
Disc Selection → [Cutie VOS Tracker] → [Post-Tracking] → Output
(initial mask)    (visual tracking)     TrackRefiner
```

- **Cutie** (CVPR 2024): Semi-supervised Video Object Segmentation model
- **Approach**: Given an initial mask from disc selection, tracks by visual appearance
- **Advantages**: 100% frame coverage, handles occlusions, no COCO class dependency
- **Output**: Same `Dict[int, List[Detection]]` format as YOLO — fully plug-and-play

Both backends use YOLO for person detection and pose estimation.

### Shared Post-Tracking

- **TrackRefiner**: Smoothing (moving average, exponential), outlier removal
- **MetricsCalculator**: Physics-based metrics (velocity, acceleration, energy, power)

## 4. Data Flow (The Hybrid Pipeline)
The system uses a configurable pipeline orchestrator (`ai-core/src/pipeline/runner.py`) driven by YAML files.

### Execution Modes
1.  **Memory Mode (Default):** Modules pass Pydantic objects directly in RAM.
2.  **Disk Mode (Debug):** Modules save/load intermediate results (JSON) to allow partial re-runs.

### Non-Linear Execution
Steps can declare inputs explicitly, allowing parallel branches:

**YOLO-only pipeline** (`configs/full_analysis.yaml`):
```
Ingestion ─┬─> YOLO COCO ──> Filter ─> Track ─> Refine ─┬─> Visualization
           └─> YOLO Pose ───────────────────────────────┘
```

**Cutie + YOLO hybrid pipeline** (`configs/cutie_analysis.yaml`):
```
Ingestion ─┬─> Cutie (disc) ────────┐
           ├─> YOLO COCO (person) ──┼─> Merger ─> Track ─> Refine ─┬─> Visualization
           └─> YOLO Pose ───────────┤                               │
                                    └───────────────────────────────┘
```

See [docs/pipeline_guide.md](pipeline_guide.md) for details on configuring runs.
