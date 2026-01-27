# Pipeline Logic & Tracking

This document explains the advanced logic used in the ai-core pipeline for object tracking, filtering, and trajectory analysis.

## 1. Non-Linear Execution Flow (Branching)

The pipeline supports "branches" where multiple models run in parallel on the same source video.

- **Ingestion**: Reads video metadata.
- **Branches**:
  - Pose Estimation -> Runs on Ingestion output.
  - Segmentation -> Runs on Ingestion output.
  - Geometric Disc Detection -> Runs on Ingestion output (HoughCircles + verification).
- **Merger**: Combines detections from all branches into a unified timeline.

## 2. Model Class Mapping (Hybrid Approach)

The pipeline combines detections from multiple sources:

1. **Custom Segmentation (YOLOv8)**: Trained on specific weightlifting objects.
   - Class 0: atleta (Athlete)
   - Class 1: barra (Barbell shaft/center)
   - Class 2: discos (Weight plates)
2. **Standard Pose Estimation (YOLOv8-Pose)**: Pre-trained on COCO Keypoints.
   - Class 0: person
3. **Geometric Disc Detector (HoughCircles + Verification)**: Classic CV with modern verification.
   - Outputs class_name="discos" but sets Detection.source="geom" and adds radius_px + shape_score.

**Integration Logic:**

- The DetectionMerger combines these streams.
- Downstream modules should use Detection.source to understand provenance without inventing new categories.
- Hybrid disc tracking uses:
  - DiscConsensusCalibrator (first N frames) to compute an initial disc prior.
  - DiscFusionTracker (Kalman + gating) to fuse YOLO + geom disc candidates.
- *Note:* Currently, the merger assumes overlapping detections are separate unless explicitly merged. The visualization draws whatever is available.

## 3. Tracking Phase (Association)

We use **ByteTrack** (or similar association logic) to link detections across frames.

- **Input**: List of Detections per frame.
- **Logic**:
  - Matches boxes based on IoU (Intersection over Union).
  - Matches low-confidence detections if they align with an existing high-confidence track (this recovers objects that briefly fade).
- **Output**: List of TrackedObject with unique IDs.

## 4. Trajectory Logic Phase (The "Cleaner")
Raw tracking data is noisy. The TrajectoryCleaner module applies biomechanical and business logic to refine the data.

### 4.1 Filtering Heuristics
We score every track to decide if it's "Relevant" or "Noise".

- **Duration Score**: Tracks shorter than N seconds are discarded.
- **Displacement Score**: Objects that don't move (e.g., a rack in the background) are discarded.
- **Centrality Score**: Objects closer to the center of the frame are weighted higher.

### 4.2 Gap Filling (Interpolation)
If an object is detected in Frame 10 and Frame 12, but missing in Frame 11:

- **Logic**: We assume the object didn't teleport.
- **Action**: We create a synthetic detection in Frame 11 at the midpoint (Linear Interpolation).
- **Advanced**: For longer gaps, we might use Kalman Filter predictions or Cubic Splines for smoother curves.

## 5. Lifting Heuristics (Business Logic)
Refer to docs/lifting_heuristics.md for details on the specific rules for Athlete and Barbell identification.

- **Active Barbell Preference**: The system prioritizes the barra class if it shows movement. If not found, it falls back to the best moving discos track.

## 6. Visualization
The final video renders:

- **IDs**: Unique numbers above objects.
- **Trails**: A history line showing where the object has been.
- **Status**: Color coding for "Active" vs "Interpolated" (predicted) detections.

## 7. Hybrid Disc Tracking Pipeline (Recommended for MVP robustness)
For the hybrid approach (YOLO + Geom + Calibration + Fusion), see the example config:

- ai-core/configs/lifting_hybrid_test.yaml
