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
│   └── /src                    <-- Production-ready Code
│       ├── /domain             <-- CONTRACTS (Entities & Interfaces)
│       ├── /input_layer        <-- Data Ingestion & ETL
│       ├── /perception         <-- Vision Models (YOLO)
│       ├── /analysis           <-- Business Logic (Filter, Track, Refine)
│       ├── /visualization      <-- Debug Tools & Renderers
│       ├── /tools              <-- Utility Tools (disc_selector)
│       └── /pipeline           <-- Orchestration
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

## 3. Three-Phase Heuristics Architecture

The pipeline separates heuristics into three phases:

```
Detection → [Pre-Tracking] → [Tracking] → [Post-Tracking] → Output
             DetectionFilter   ModelTracker   TrackRefiner
```

### Phase 1: Pre-Tracking (DetectionFilter)
- **State**: None (stateless, per-frame)
- **Purpose**: Reduce candidate detections
- **Filters**: Size, confidence, ROI, largest-selector

### Phase 2: Tracking (ModelTracker)
- **State**: Temporal (Kalman filter)
- **Purpose**: Assign persistent IDs to objects
- **Features**: Hungarian assignment, track lifecycle, single-object mode

### Phase 3: Post-Tracking (TrackRefiner)
- **State**: Full trajectory
- **Purpose**: Refine trajectories
- **Features**: Smoothing (moving average, exponential), outlier removal

## 4. Data Flow (The Hybrid Pipeline)
The system uses a configurable pipeline orchestrator (`ai-core/src/pipeline/runner.py`) driven by YAML files.

### Execution Modes
1.  **Memory Mode (Default):** Modules pass Pydantic objects directly in RAM.
2.  **Disk Mode (Debug):** Modules save/load intermediate results (JSON) to allow partial re-runs.

### Non-Linear Execution
Steps can declare inputs explicitly, allowing parallel branches:
*   `YOLO Custom` consumes `Ingestion` output.
*   `YOLO COCO` consumes `Ingestion` output.
*   `Multi-Model Renderer` consumes multiple model outputs.

```
Ingestion ─┬─> YOLO Custom ─> Filter ─> Track ─> Refine ─┬─> Visualization
           ├─> YOLO COCO ──> Filter ─> Track ─> Refine ──┤
           └─> YOLO Pose ─────────────────────────────────┘
```

See [docs/pipeline_guide.md](pipeline_guide.md) for details on configuring runs.
