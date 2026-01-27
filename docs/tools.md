
## Visualization (`video_renderer.py`)

### Purpose
Generates debugging videos by overlaying the detection results (masks, boxes, labels) onto the original footage. It uses color coding for different classes and transparency for segmentation masks.

### Configuration
In your YAML pipeline config:
```yaml
  - name: visualization
    module: "video_renderer"
    enabled: true
    input_source: "memory" # Receives detections from Perception step
    params:
      video_source: "data/raw/video_test_1.mp4" # Path to the original video
      output_filename: "overlay.mp4" # Optional, defaults to overlay.mp4
```

### Output
The module generates an MP4 video file in the run's output directory (e.g., `data/outputs/{run_id}/overlay.mp4`).
