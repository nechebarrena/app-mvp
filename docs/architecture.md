# Project Architecture

## Overview
This project is designed as an MVP for a mobile video analysis application. The core logic is developed in Python ("AI Core") with the intention of exporting optimized models to a native Mobile App.

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
│   ├── /outputs                <-- Final results (JSON reports, Overlay videos)
│   └── /models                 <-- Model weights (YOLO .pt, .tflite, .onnx)
│
├── /ai-core                    <-- PYTHON CORE
│   ├── pyproject.toml          <-- Dependencies (managed by uv)
│   ├── /notebooks              <-- Research & Sandboxing
│   └── /src                    <-- Production-ready Code
│       ├── /domain             <-- CONTRACTS (Entities & Interfaces)
│       ├── /input_layer        <-- Data Ingestion & ETL
│       ├── /perception         <-- Vision Models (YOLO)
│       ├── /analysis           <-- Business Logic (Tracking, Physics)
│       ├── /visualization      <-- Debug Tools & Dashboards
│       └── /pipeline           <-- Orchestration
│
└── /mobile-app                 <-- MOBILE CLIENT (Flutter/React Native)
```

## 2. Domain-Driven Design (DDD) & Contracts
To ensure modularity and testability, we use a simplified DDD approach located in `ai-core/src/domain`.

### Entities (`entities.py`)
Data structures that flow between modules. They are "dumb" data containers (Pydantic models or Dataclasses).
*   **VideoSession:** Represents a loaded video file and its metadata.
*   **FrameData:** A single frame image with timestamp.
*   **Detection:** Result of the generic object detector (bounding box, class, mask, keypoints).
*   **TrackedObject:** Result of the tracker (ID, history of positions).

### Ports (`ports.py`)
Interfaces (Python Protocols) that define what a module *must do*, decoupling the *what* from the *how*.
*   **IPipelineStep:** Generic interface for any processing step, enforcing I/O capabilities.

## 3. Data Flow (The Hybrid Pipeline)
The system uses a configurable pipeline orchestrator (`ai-core/src/pipeline/runner.py`) driven by YAML files.

### Execution Modes
1.  **Memory Mode (Default):** Modules pass Pydantic objects directly in RAM.
2.  **Disk Mode (Debug):** Modules save/load intermediate results (JSON) to allow partial re-runs.

### Non-Linear Execution
Steps can declare inputs explicitly, allowing parallel branches:
*   `Pose Estimation` consumes `Ingestion` output.
*   `Segmentation` consumes `Ingestion` output.
*   `Merger` consumes both `Pose` and `Segmentation` outputs.
*   `Visualization` consumes `Merger` output.

See [docs/pipeline_guide.md](docs/pipeline_guide.md) for details on configuring runs.
