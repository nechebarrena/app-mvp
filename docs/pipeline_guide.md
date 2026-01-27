# The Hybrid Pipeline

## Overview
The project uses a configurable pipeline architecture that supports hybrid data flow: **Memory** for speed (mimicking the mobile app) and **Disk** for debugging/persistence.

### Key Concepts
1.  **Orchestration via YAML:** Every run is defined by a configuration file.
2.  **IPipelineStep Interface:** All modules (Input, Perception, Analysis) must implement:
    *   `run(input, config)`: Core logic.
    *   `save_result(data, path)`: Serialize to disk.
    *   `load_result(path)`: Deserialize from disk.
3.  **Logging:** Every run generates a `pipeline.log` ensuring full traceability.

## Memory vs. Disk Mode
*   **Memory Mode (Production/Mobile):** Steps pass Python objects (Pydantic entities) directly.
    *   *Step 1 (Frame)* -> *Step 2 (Detections)* -> *Step 3 (Tracks)*
*   **Disk Mode (Development/Debug):** Steps serialize their output to JSON/CSV.
    *   *Step 1* -> `frames/` -> *Step 2* -> `detections.json` -> *Step 3*

This allows developers to "resume" a pipeline from the middle. For example, if you are tweaking the Tracker, you don't need to re-run YOLO (Perception). You just load the detections.json from the previous run.

## Configuration Schema (pipeline_config.yaml)

session:
  video_id: "video_001"
  output_dir: "debug_run_1"

steps:
  - name: my_step
    module: "registered_module_name"
    enabled: true
    input_source: "memory" # or "disk"
    save_output: true
    params:
      threshold: 0.5

## Example: Hybrid Disc Tracking (YOLO + Geometry + Calibration + Fusion)

See the ready-to-run config:
- ai-core/configs/lifting_hybrid_test.yaml

This pipeline runs three perception branches in parallel:
- yolo_segmentor (segmentation; Detection.source="yolo")
- yolo_pose (pose; Detection.source="pose")
- geom_circle_detector (HoughCircles + verification; Detection.source="geom")

Then it performs:
- disc_calibrator: builds disc_prior.json from consensus in the first calibration_frames frames
- disc_fusion_tracker: Kalman + gating + multi-source fusion, producing:
  - disc_tracker_debug.jsonl (per-frame)
  - disc_tracker_debug_summary.json (run summary)
