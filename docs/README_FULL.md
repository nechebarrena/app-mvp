# APP-MVP: AI Video Analysis - Complete Documentation

> **Generated:** January 31, 2026  
> **Version:** 0.6.0 (Control Panel + Ngrok)
> **Purpose:** Complete reference document for AI review and project understanding

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Three-Phase Heuristics Architecture](#4-three-phase-heuristics-architecture)
5. [Pipeline System](#5-pipeline-system)
6. [Available Modules](#6-available-modules)
7. [Metrics Calculation System](#7-metrics-calculation-system)
8. [Interactive Analysis Viewer](#8-interactive-analysis-viewer)
9. [Control Panel (Web Dashboard)](#9-control-panel-web-dashboard)
10. [FastAPI Backend](#10-fastapi-backend)
11. [Model Management](#11-model-management)
12. [Label Mapping System](#12-label-mapping-system)
13. [Configuration Examples](#13-configuration-examples)
14. [Running the Pipeline](#14-running-the-pipeline)
15. [Current Status & Capabilities](#15-current-status--capabilities)

---

## 1. Project Overview

### Goal
Build an MVP for mobile video analysis that:
1. Captures short video clips (10-60s) of weightlifting sessions
2. Processes them using AI (Object Detection + Segmentation + Pose Estimation)
3. Tracks objects of interest (disc, athlete, barbell)
4. Applies domain heuristics (single athlete, single disc, size constraints)
5. **Calculates physical metrics** (position, velocity, acceleration, energy, power)
6. **Provides interactive visualization** for analysis and debugging
7. Displays results to the user

### Architecture Philosophy
- **Python AI Core**: R&D lab and processing engine
- **Modular Pipeline**: YAML-configured, plug-and-play modules
- **Multi-Model Support**: Run multiple YOLO models in parallel
- **Three-Phase Heuristics**: Pre-tracking → Tracking → Post-tracking separation
- **Label Unification**: Map different model outputs to unified concepts
- **Metrics Engine**: Calculate physics-based metrics from tracked trajectories
- **Interactive Tools**: GUI tools for selection, analysis, and debugging

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
   - **Variable Substitution**: YAML variables for DRY configuration

3. **Multi-Model Fusion**
   - Run detection, segmentation, and pose models in parallel
   - Label mapping for concept unification across models

4. **Three-Phase Heuristics**
   - Pre-tracking: Stateless filters (size, confidence, largest-selector)
   - Tracking: Temporal association (Kalman, Hungarian)
   - Post-tracking: Trajectory refinement (smoothing, outlier removal)

5. **Metrics & Visualization** (NEW)
   - Physics-based metrics calculation from tracked trajectories
   - Interactive GUI for synchronized video + graph analysis

---

## 3. Directory Structure

```
/app-mvp
├── /ai-core                    # PYTHON CORE
│   ├── /configs                # Pipeline YAML configurations
│   │   ├── full_analysis.yaml       # Complete pipeline with metrics
│   │   ├── single_disc_tracking.yaml # 3-phase tracking pipeline
│   │   ├── compare_models.yaml       # Multi-model comparison
│   │   ├── physical_params.yaml      # Physical parameters for metrics
│   │   └── ...
│   ├── /models                 # Model weights
│   │   ├── /custom             # Custom trained models
│   │   │   └── best.pt         # Weightlifting segmentation (atleta, barra, discos)
│   │   └── /pretrained         # Standard pretrained models
│   │       ├── yolov8s-seg.pt  # COCO segmentation small (24MB)
│   │       ├── yolov8n-pose.pt # COCO pose estimation
│   │       └── yolov8n-seg.pt  # COCO segmentation nano
│   ├── /src
│   │   ├── /domain             # Core contracts
│   │   │   ├── entities.py     # Detection, TrackedObject, VideoSession
│   │   │   ├── ports.py        # IPipelineStep interface
│   │   │   └── label_mapper.py # Label unification system
│   │   ├── /input_layer        # Data ingestion
│   │   │   ├── video_loader.py
│   │   │   └── metadata_scanner.py  # Generate video metadata JSON
│   │   ├── /perception         # AI models
│   │   │   ├── yolo_detector.py    # Generic YOLO (detect/segment/pose)
│   │   │   ├── yolo_segmentor.py   # Legacy segmentation
│   │   │   └── yolo_pose.py        # Legacy pose
│   │   ├── /analysis           # Business logic
│   │   │   ├── detection_filter.py   # PRE-TRACKING heuristics
│   │   │   ├── model_tracker.py      # TRACKING (Kalman + Hungarian)
│   │   │   ├── track_refiner.py      # POST-TRACKING refinement
│   │   │   ├── metrics_calculator.py # Physical metrics calculation (NEW)
│   │   │   ├── merger.py             # Detection merger
│   │   │   └── lifting_optimizer.py  # Domain-specific logic
│   │   ├── /visualization      # Output renderers
│   │   │   ├── multi_model_renderer.py  # Multi-panel comparison video
│   │   │   ├── video_renderer.py        # Single overlay video
│   │   │   ├── metrics_visualizer.py    # Static metrics plots (NEW)
│   │   │   └── interactive_viewer.py    # Interactive GUI viewer (NEW)
│   │   ├── /tools              # Utility tools
│   │   │   ├── disc_selector.py      # GUI for manual disc selection
│   │   │   └── selection_loader.py   # Load selection into pipeline
│   │   └── /pipeline           # Orchestration
│   │       ├── runner.py       # Pipeline executor (variable substitution)
│   │       └── config.py       # Configuration models
│   ├── run_pipeline.py         # Main entry point
│   ├── run_api.py              # FastAPI server launcher
│   ├── control_panel.py        # Web dashboard (NEW)
│   ├── select_disc.py          # Manual disc selection tool
│   ├── view_analysis.py        # Interactive viewer launcher
│   ├── test_api_full.py        # API test script
│   ├── /templates              # HTML templates for control panel
│   └── pyproject.toml          # Dependencies (uv)
│
├── /data                       # DATA STORAGE (gitignored)
│   ├── /raw                    # Input videos
│   ├── /outputs                # Pipeline outputs per run
│   │   ├── disc_selection.json # Manual disc selection
│   │   └── full_analysis_run/  # Example output
│   │       ├── metrics_calculator_output.csv
│   │       ├── metrics_plot.png
│   │       └── tracking_video.mp4
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

---

## 5. Pipeline System

### Configuration Schema with Variables

```yaml
# YAML Variables (DRY principle)
variables:
  video_name: "video_test_1"
  output_name: "full_analysis_run"

# Disc selection (interactive or file-based)
disc_selection:
  mode: "file"  # "interactive" or "file"
  output_file: "../data/outputs/disc_selection.json"

session:
  video_id: "${video_name}"      # Uses variable substitution
  output_dir: "${output_name}"

steps:
  - name: step_name
    module: "registered_module"
    enabled: true
    input_source: "memory"
    input_from_step: "previous"
    save_output: true
    params:
      key: value
```

### Execution Flow (Full Analysis)

```
Ingestion ─┬─> YOLO COCO ─> Filter ─> Track ─> Refine ─┬─> Tracking Video
           └─> YOLO Pose ─────────────────────────────┘         │
                                                                 │
                                                                 v
                                              Metrics Calculator ─> Metrics Plot
                                                                 │
                                                                 v
                                                       Interactive Viewer
```

---

## 6. Available Modules

### Input Layer
| Module | Class | Description |
|--------|-------|-------------|
| `video_loader` | VideoLoader | Loads video, creates VideoSession, extracts metadata |
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
| `metrics_calculator` | MetricsCalculator | **Metrics** | Physics calculations |

### Visualization
| Module | Class | Description |
|--------|-------|-------------|
| `multi_model_renderer` | MultiModelRenderer | **Multi-panel comparison video** |
| `video_renderer` | VideoOverlayRenderer | Single overlay video |
| `metrics_visualizer` | MetricsVisualizer | Static plots (PNG) |
| `interactive_viewer` | InteractiveViewerLauncher | **Interactive GUI** |

### Tools
| Tool | Description |
|------|-------------|
| `select_disc.py` | GUI for manual disc center/radius selection |
| `view_analysis.py` | Launch interactive viewer for existing run |

---

## 7. Metrics Calculation System

### Overview

The `MetricsCalculator` module computes physical metrics from tracked disc trajectories.

### Input Requirements

1. **Tracked Objects**: Output from `TrackRefiner` with disc trajectory
2. **Physical Parameters** (from `physical_params.yaml`):
   - `disc_diameter_m`: Disc diameter in meters (for scale calculation)
   - `disc_weight_kg`: Weight per disc
   - `bar_weight_kg`: Barbell weight
   - `num_discs`: Number of discs (typically 2)
3. **Video Metadata** (auto-propagated): FPS, resolution, duration

### Output Metrics

| Metric | Column | Unit | Description |
|--------|--------|------|-------------|
| Frame | `frame_idx` | - | Frame number |
| Time | `time_s` | s | Timestamp |
| Position X | `x_m` | m | Horizontal position |
| Position Y | `y_m` | m | Vertical position (inverted) |
| Height | `height_m` | m | Height above minimum |
| Velocity X | `vx_m_s` | m/s | Horizontal velocity |
| Velocity Y | `vy_m_s` | m/s | Vertical velocity |
| Speed | `speed_m_s` | m/s | Total speed magnitude |
| Accel X | `ax_m_s2` | m/s² | Horizontal acceleration |
| Accel Y | `ay_m_s2` | m/s² | Vertical acceleration |
| Accel | `accel_m_s2` | m/s² | Total acceleration |
| Kinetic E | `kinetic_energy_j` | J | ½mv² |
| Potential E | `potential_energy_j` | J | mgh |
| Total E | `total_energy_j` | J | KE + PE |
| Power | `power_w` | W | F·v (force × velocity) |

### Scale Calculation

The pixel-to-meter scale is calculated from the disc selection:
```python
scale_m_per_px = disc_diameter_m / (2 * reference_radius_px)
```

### Example Configuration

```yaml
metrics_calculator:
  module: "metrics_calculator"
  input_from_step: "track_refiner"
  params:
    target_class: "discos"
    physical_params_file: "configs/physical_params.yaml"
```

---

## 8. Interactive Analysis Viewer

### Overview

The `InteractiveAnalysisViewer` is a PyQt5-based GUI that provides synchronized video and metrics visualization.

**Unified Data Loader**: The viewer supports multiple input sources:
- **Pipeline output**: CSV metrics + video with tracking overlays
- **API JSON**: results.json + original video

### Features

1. **3-Panel Layout**
   - Left: Selectable graph
   - Center: Video with playback controls
   - Right: Selectable graph (default: X-Y Trajectory)

2. **Video Controls**
   - ▶ Play / ⏸ Pause
   - ⏹ Stop (return to start)
   - 🐢 0.25x slow motion
   - Frame-by-frame navigation (±1, ±10)

3. **Video Trimming**
   - Drag green/red markers on slider
   - "Marcar Inicio" / "Marcar Fin" buttons
   - "Resetear Recorte" to restore full video
   - Graphs automatically update to trimmed range

4. **Available Graphs**
   | Graph | Type |
   |-------|------|
   | Altura (m) | Time series |
   | Velocidad (m/s) | Time series |
   | Velocidad Y (m/s) | Time series |
   | Aceleración (m/s²) | Time series |
   | Energía Cinética (J) | Time series |
   | Energía Potencial (J) | Time series |
   | Energía Total (J) | Time series |
   | Potencia (W) | Time series |
   | Posición X (m) | Time series |
   | Posición Y (m) | Time series |
   | **📍 Trayectoria X-Y** | **Trajectory plot** |

5. **Trajectory Plot Options**
   - Toggle velocity colormap
   - Line thickness (1-10)
   - Colormap selection (plasma, viridis, inferno, coolwarm, etc.)
   - Current position marker synced with video

### Launching

```bash
# From pipeline output directory
cd ai-core
PYTHONPATH=src:. uv run python view_analysis.py ../data/outputs/full_analysis_run

# From API results.json
PYTHONPATH=src:. uv run python view_analysis.py ../data/api/results/{video_id}/results.json

# With explicit video path
PYTHONPATH=src:. uv run python view_analysis.py results.json --video /path/to/video.mp4

# Via pipeline (auto-launches after metrics calculation)
# Set in YAML:
#   - name: interactive_viewer
#     module: "interactive_viewer"
```

### Data Loader (`visualization/data_loader.py`)

```python
# Auto-detect source type
from visualization.data_loader import load_viewer_data

# From directory (auto-detects pipeline or API)
data = load_viewer_data("../data/outputs/full_analysis_run")

# From API JSON
data = load_viewer_data("results.json", video_path="/path/to/video.mp4")

# Returns ViewerData with: video_path, metrics_df, fps, metadata
```

---

## 9. Control Panel (Web Dashboard)

A web-based interface for easy system operation without terminal commands.

### Usage

```bash
cd ai-core
PYTHONPATH=src:. uv run python control_panel.py
# Opens browser automatically at http://localhost:5001
```

### Features

| Feature | Description |
|---------|-------------|
| **Service Control** | Start/Stop FastAPI and Ngrok with buttons |
| **Status Indicators** | Visual indicators for running services |
| **Video Selection** | Browse and select videos from data/raw/ |
| **Disc Selector** | Open GUI tool to select disc position |
| **Processing** | One-click video processing with progress bar |
| **Results** | View summary and open interactive viewer |
| **Logs** | Real-time log display |

### Architecture

```
control_panel.py (Flask :5001)
        │
        ├── Controls → FastAPI (:8000)
        ├── Controls → Ngrok tunnel
        ├── Launches → select_disc.py (PyQt5)
        └── Launches → view_analysis.py (PyQt5)
```

The Control Panel manages all other services as subprocesses.

---

## 10. FastAPI Backend

### Overview

The FastAPI backend provides REST endpoints for mobile app integration. It implements
the **Client-Side Rendering** architecture where:

1. Mobile uploads video once (with optional disc selection parameters)
2. Server processes with existing pipeline
3. Server returns lightweight JSON (~200KB vs ~50MB video)
4. Mobile renders overlays locally

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/videos/upload` | Upload video for processing |
| `GET` | `/api/v1/videos/{id}/status` | Check processing status |
| `GET` | `/api/v1/videos/{id}/results` | Get analysis results |
| `DELETE` | `/api/v1/videos/{id}` | Delete video and results |
| `GET` | `/api/v1/videos` | List all videos |

### Upload with Disc Selection

The upload endpoint supports optional disc selection parameters for better tracking:

```bash
curl -X POST "http://localhost:8000/api/v1/videos/upload" \
  -F "file=@video.mp4" \
  -F "disc_center_x=587" \
  -F "disc_center_y=623" \
  -F "disc_radius=74"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | File | Video file (required) |
| `disc_center_x` | float | Disc center X in frame 0 (optional) |
| `disc_center_y` | float | Disc center Y in frame 0 (optional) |
| `disc_radius` | float | Disc radius in pixels (optional) |

When disc selection is provided, the pipeline activates:
- **initial_selector**: Picks the detection closest to the reference in frame 0
- **single-object tracking**: Maintains only one track for the disc class

### Response Format

```json
{
  "video_id": "abc123",
  "metadata": { "fps": 30, "width": 1080, "height": 1920, ... },
  "tracks": [
    {
      "track_id": 1,
      "class_name": "disco",
      "frames": { "0": { "bbox": {...}, "mask": [...] }, ... },
      "trajectory": [[x, y], ...]
    }
  ],
  "metrics": {
    "frames": [0, 1, 2, ...],
    "height_m": [0.5, 0.52, ...],
    "speed_m_s": [0.0, 0.6, ...],
    "power_w": [0, 150, ...]
  },
  "summary": {
    "peak_speed_m_s": 4.02,
    "peak_power_w": 3827
  }
}
```

### Running the Server

```bash
cd ai-core
PYTHONPATH=src:. uv run python run_api.py --reload
```

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Test Script

A complete test script simulates the mobile app workflow:

```bash
# Terminal 1: Start server
cd ai-core
PYTHONPATH=src:. uv run python run_api.py

# Terminal 2: Run test with disc selection GUI
cd ai-core
PYTHONPATH=src:. uv run python test_api_full.py --video ../data/raw/video_test_1.mp4
```

The test script:
1. Verifies the server is running
2. Opens the interactive disc selection tool
3. Uploads the video with selection coordinates
4. Polls for processing completion
5. Displays the results

### Visualizing API Results

The interactive viewer supports both pipeline output and API JSON:

```bash
# From pipeline output
PYTHONPATH=src:. uv run python view_analysis.py ../data/outputs/full_analysis_run/

# From API JSON
PYTHONPATH=src:. uv run python view_analysis.py ../data/api/results/{video_id}/results.json
```

### Remote Access with Ngrok

For accessing the server from anywhere (not just local network):

```bash
# Terminal 1: FastAPI server
cd ai-core && PYTHONPATH=src:. uv run python run_api.py

# Terminal 2: Ngrok tunnel
ngrok http 8000
# Provides URL like: https://abc123.ngrok-free.app
```

Mobile app (or any client) can then access:
- `https://abc123.ngrok-free.app/api/v1/videos/upload`
- `https://abc123.ngrok-free.app/api/v1/videos/{id}/status`
- `https://abc123.ngrok-free.app/api/v1/videos/{id}/results`

**Note:** Include header `ngrok-skip-browser-warning: true` in all requests.

### Architecture Benefits

| Aspect | Value |
|--------|-------|
| Bandwidth | ~50MB (upload only) vs ~100MB (upload + download) |
| Flexibility | Mobile can customize rendering |
| Caching | JSON easily cached |
| Offline Path | Same render code can use on-device inference |

See [API Guide](api_guide.md) and [API_README.md](../ai-core/API_README.md) for complete documentation.

---

## 11. Model Management

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

### Known Limitations

- **Black discs**: COCO "frisbee" class doesn't generalize well to black weightlifting discs
- **Recommendation**: Use custom-trained model for disc detection, or improve training data

---

## 12. Label Mapping System

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

## 13. Configuration Examples

### Full Analysis Pipeline

```yaml
# configs/full_analysis.yaml
variables:
  video_name: "video_test_1"
  output_name: "full_analysis_run"

disc_selection:
  mode: "file"  # or "interactive"
  output_file: "../data/outputs/disc_selection.json"

session:
  video_id: "${video_name}"
  output_dir: "${output_name}"

steps:
  # Stage 0: Video Loading
  - name: ingestion
    module: "video_loader"

  # Stage 1: Detection
  - name: yolo_coco_detection
    module: "yolo_detector"
    params:
      model_path: "models/pretrained/yolov8s-seg.pt"
      task: "segment"
      source_name: "yolo_coco"

  # Stage 2: Pre-Tracking Filter
  - name: detection_filter
    module: "detection_filter"
    params:
      size_filter:
        enabled: true
        classes: ["frisbee"]
        tolerance: 0.30
      largest_selector:
        enabled: true
        classes: ["person"]

  # Stage 3: Tracking
  - name: disc_tracking
    module: "model_tracker"
    params:
      classes_to_track: ["frisbee", "person"]
      single_object_classes: ["frisbee"]

  # Stage 4: Post-Tracking Refinement
  - name: track_refiner
    module: "track_refiner"
    params:
      smoothing:
        enabled: true
        method: "moving_average"
        window: 5

  # Stage 5: Video Output
  - name: tracking_video
    module: "multi_model_renderer"

  # Stage 6: Metrics Calculation
  - name: metrics_calculator
    module: "metrics_calculator"
    params:
      target_class: "frisbee"
      physical_params_file: "configs/physical_params.yaml"

  # Stage 7: Static Plots
  - name: metrics_visualizer
    module: "metrics_visualizer"

  # Stage 8: Interactive Viewer
  - name: interactive_viewer
    module: "interactive_viewer"
    params:
      title: "Análisis de Levantamiento"
```

---

## 14. Running the Pipeline

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
cd ai-core
uv sync
```

### Step 1: Manual Disc Selection (if needed)

```bash
cd ai-core
PYTHONPATH=src:. uv run python select_disc.py
```

### Step 2: Run Full Pipeline

```bash
cd ai-core
PYTHONPATH=src:. uv run python run_pipeline.py configs/full_analysis.yaml
```

### Step 3: View Results Interactively

```bash
cd ai-core
PYTHONPATH=src:. uv run python view_analysis.py full_analysis_run
```

### Output Structure

```
data/outputs/full_analysis_run/
├── pipeline.log                           # Execution log
├── yolo_coco_detection_output.json        # Raw detections
├── detection_filter_output.json           # After pre-filter
├── disc_tracking_output.json              # After tracking
├── disc_tracking_output_summary.json      # Track statistics
├── track_refiner_output.json              # After refinement
├── tracking_video.mp4                     # Video with overlays
├── tracking_video_output.json             # Video metadata
├── metrics_calculator_output.csv          # Metrics data
├── metrics_calculator_output.json         # Metrics metadata
├── metrics_plot.png                       # Static metrics plots
└── metrics_visualizer_output.json         # Plot metadata
```

---

## 15. Current Status & Capabilities

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

6. **Metrics Calculation**
   - Position, velocity, acceleration
   - Kinetic, potential, total energy
   - Power calculation
   - Automatic scale from disc size

7. **Interactive Analysis Viewer**
   - Synchronized video + graphs
   - Video trimming with dynamic graph updates
   - Trajectory X-Y plot with velocity colormap
   - Play/pause/slow-motion controls

8. **FastAPI Backend**
   - REST API for mobile app integration
   - Async video processing with progress tracking
   - Client-side rendering architecture (JSON response ~200KB)
   - Swagger UI documentation at /docs
   - Disc selection parameters for improved tracking

9. **Control Panel (Web Dashboard)** (NEW)
   - One-command startup: `python control_panel.py`
   - Web interface at http://localhost:5001
   - Start/Stop FastAPI and Ngrok with buttons
   - Video selection and disc selector integration
   - Processing progress and results display
   - Real-time logs

10. **YAML Variable Substitution**
    - DRY configuration with `${variable}` syntax
    - Video metadata auto-propagation

### 📋 Future Work

1. **Improved Detection**
   - Better training data for black discs
   - Custom YOLO model refinement
   - Multi-disc handling

2. **Advanced Metrics**
   - Rep counting
   - Velocity-based load estimation
   - Fatigue detection

3. **Mobile App Development** (See `MOBILE_APP_SPEC.md`)
   - Kotlin Multiplatform (Android first, iOS later)
   - Client-side overlay rendering
   - Disc selection on device
   - Offline capability (Phase 2)

4. **Mobile Export**
   - CoreML / TFLite conversion
   - Optimized on-device inference

---

## 16. Mobile App Integration

The system is designed for mobile app integration. See **`docs/MOBILE_APP_SPEC.md`** for complete specification.

### Quick Reference

**Server Setup:**
```bash
cd ai-core
PYTHONPATH=src:. uv run python control_panel.py
# Start FastAPI + Ngrok from the web UI
```

**Mobile → Server Contract:**
```
POST /api/v1/videos/upload
  - file: video/mp4
  - disc_center_x, disc_center_y, disc_radius (optional)

GET /api/v1/videos/{id}/status
  - Poll every 2-3 seconds

GET /api/v1/videos/{id}/results
  - JSON with tracks, metrics, summary
```

**Key Headers:**
```
ngrok-skip-browser-warning: true
Content-Type: multipart/form-data
```

---

## Appendix: Entity Schemas

### Detection

```python
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    mask: Optional[List[List[float]]]        # Polygon points
    keypoints: Optional[List[List[float]]]   # [[x, y, conf], ...]
    source: Optional[str]                    # "yolo_custom", "yolo_coco", etc.
```

### TrackedObject

```python
class TrackedObject(BaseModel):
    track_id: int
    detection: Detection
    history: List[Tuple[float, float]]       # Center points
    velocity: Optional[Tuple[float, float]]
    smoothed_position: Optional[Tuple[float, float]]  # After refinement
    smoothed_history: List[Tuple[float, float]]       # Full smoothed trajectory
```

### Metrics DataFrame

```python
# Columns in metrics_calculator_output.csv
columns = [
    'frame_idx', 'time_s',
    'x_m', 'y_m', 'height_m',
    'vx_m_s', 'vy_m_s', 'speed_m_s',
    'ax_m_s2', 'ay_m_s2', 'accel_m_s2',
    'kinetic_energy_j', 'potential_energy_j', 'total_energy_j',
    'power_w'
]
```

---

*End of Documentation*
