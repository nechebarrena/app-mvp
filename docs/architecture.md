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
*   **Detection:** Result of the generic object detector (bounding box, class).
*   **TrackedObject:** Result of the tracker (ID, history of positions).

### Ports (`ports.py`)
Interfaces (Python Protocols) that define what a module *must do*, decoupling the *what* from the *how*.
*   **IVideoLoader:** Defines how to load video sessions.
*   **IObjectDetector:** Defines how to detect objects in a frame.
*   **ITracker:** Defines how to associate detections over time.

## 3. Dependency Management
We use **`uv`** for lightning-fast package management.
*   **Configuration:** All dependencies are listed in `ai-core/pyproject.toml`.
*   **Lockfile:** `ai-core/uv.lock` ensures reproducible builds.
*   **Virtual Environment:** Automatically managed by `uv` in `.venv`.

## 4. Data Flow (The Pipeline)
1.  **Ingestion:** `Input Layer` reads `.mp4` + `.json` sidecar -> Produces `FrameData`.
2.  **Perception:** `Perception Module` (YOLO) consumes `FrameData` -> Produces `List[Detection]`.
3.  **Analysis:** `Analysis Module` consumes `List[Detection]` -> Produces `List[TrackedObject]` and metrics.
4.  **Output:** `Visualization` renders results to disk or screen.

