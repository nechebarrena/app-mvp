# The Hybrid Pipeline

## Overview
The project uses a configurable pipeline architecture that supports hybrid data flow: **Memory** for speed (mimicking the mobile app) and **Disk** for debugging/persistence.

### Key Concepts
1.  **Orchestration via YAML:** Every run is defined by a configuration file.
2.  **IPipelineStep Interface:** All modules must implement:
    *   `run(input, config)`: Core logic.
    *   `save_result(data, path)`: Serialize to disk.
    *   `load_result(path)`: Deserialize from disk.
3.  **Logging:** Every run generates a `pipeline.log` ensuring full traceability.

## Memory vs. Disk Mode
*   **Memory Mode (Production/Mobile):** Steps pass Python objects (Pydantic entities) directly.
*   **Disk Mode (Development/Debug):** Steps serialize their output to JSON.

This allows developers to "resume" a pipeline from the middle.

## Configuration Schema

```yaml
session:
  video_id: "video_001"
  output_dir: "debug_run_1"

steps:
  - name: my_step
    module: "registered_module_name"
    enabled: true
    input_source: "memory"
    input_from_step: "previous_step"
    save_output: true
    params:
      threshold: 0.5
```

## Three-Phase Tracking Pipeline

The main pipeline follows this structure:

```
Ingestion -> Detection -> Filter -> Track -> Refine -> Visualization
```

### Main Config: single_disc_tracking.yaml

This pipeline demonstrates the three-phase heuristics architecture:

1. **Detection**: YOLO models (custom + COCO + pose)
2. **Pre-Tracking Filter**: Size filter, largest-selector for athlete
3. **Tracking**: Kalman + Hungarian with single-object mode
4. **Post-Tracking Refine**: Moving average smoothing
5. **Visualization**: Multi-panel comparison video

## Running a Pipeline

```bash
cd ai-core

# Step 1: Select disc (manual)
PYTHONPATH=src:. uv run python select_disc.py

# Step 2: Run pipeline
PYTHONPATH=src:. uv run python run_pipeline.py configs/single_disc_tracking.yaml
```

## Available Modules

| Module | Description |
|--------|-------------|
| `video_loader` | Load video files |
| `yolo_detector` | Generic YOLO (detect/segment/pose) |
| `detection_filter` | Pre-tracking heuristics |
| `model_tracker` | Kalman + Hungarian tracking |
| `track_refiner` | Post-tracking smoothing |
| `multi_model_renderer` | Multi-panel comparison video |
| `selection_loader` | Load manual disc selection |
