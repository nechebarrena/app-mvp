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
4.  **Variable Substitution:** YAML supports `${variable}` syntax for DRY configuration.

## Memory vs. Disk Mode
*   **Memory Mode (Production/Mobile):** Steps pass Python objects (Pydantic entities) directly.
*   **Disk Mode (Development/Debug):** Steps serialize their output to JSON.

This allows developers to "resume" a pipeline from the middle.

## Configuration Schema

```yaml
# Variables for DRY configuration
variables:
  video_name: "video_test_1"
  output_name: "my_run"

# Disc selection config
disc_selection:
  mode: "file"  # or "interactive" to launch GUI
  output_file: "../data/outputs/disc_selection.json"

session:
  video_id: "${video_name}"   # Uses variable
  output_dir: "${output_name}"

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

## Full Analysis Pipeline

The complete pipeline includes detection, tracking, metrics, and visualization:

```
Ingestion -> Detection -> Filter -> Track -> Refine -> Video -> Metrics -> Viewer
```

### Main Config: full_analysis.yaml

This pipeline demonstrates the full workflow:

1. **Ingestion**: Load video, extract metadata (FPS, resolution)
2. **Detection**: YOLO models (custom + COCO + pose)
3. **Pre-Tracking Filter**: Size filter, largest-selector for athlete
4. **Tracking**: Kalman + Hungarian with single-object mode
5. **Post-Tracking Refine**: Moving average smoothing
6. **Visualization**: Multi-panel tracking video
7. **Metrics Calculation**: Position, velocity, acceleration, energy, power
8. **Metrics Plot**: Static PNG with all metrics
9. **Interactive Viewer**: GUI for synchronized video + graph analysis

## Running a Pipeline

```bash
cd ai-core

# Step 1: Select disc (manual) - only needed once per video
PYTHONPATH=src:. uv run python select_disc.py

# Step 2: Run full analysis pipeline
PYTHONPATH=src:. uv run python run_pipeline.py configs/full_analysis.yaml

# Step 3: View results interactively (if not auto-launched)
PYTHONPATH=src:. uv run python view_analysis.py full_analysis_run
```

## Available Modules

| Module | Description |
|--------|-------------|
| `video_loader` | Load video files, extract metadata |
| `yolo_detector` | Generic YOLO (detect/segment/pose) |
| `detection_filter` | Pre-tracking heuristics (size, largest-selector) |
| `model_tracker` | Kalman + Hungarian tracking |
| `track_refiner` | Post-tracking smoothing |
| `multi_model_renderer` | Multi-panel comparison video |
| `metrics_calculator` | Physics metrics from trajectory |
| `metrics_visualizer` | Static plots (PNG) |
| `interactive_viewer` | Interactive GUI analysis |
| `selection_loader` | Load manual disc selection |

## Video Metadata Propagation

The pipeline automatically extracts and propagates video metadata:

- `_video_fps`: Frames per second
- `_video_width`, `_video_height`: Resolution
- `_video_duration`: Total duration in seconds
- `_video_total_frames`: Total frame count

These are injected into all step parameters and can be used by modules like `metrics_calculator`.
