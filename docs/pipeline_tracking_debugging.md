# Pipeline / Tracking Debugging Guide

This document explains how data flows through the pipeline (classes, IDs, merges) and where tracking failures commonly originate.

## 1. Data Flow (high-level)

1. **Ingestion (video_loader)**
   - Creates a VideoSession (path, camera specs, etc.).

2. **Perception (Segmentation: yolo_segmentor)**
   - Outputs Dict[frame_idx, List[Detection]].
   - Detection.class_name comes from the **custom model** (best.pt): atleta, barra, discos.
   - Detection.source="yolo".

3. **Perception (Pose: yolo_pose)**
   - Outputs Dict[frame_idx, List[Detection]].
   - Detection.class_name comes from the **pose model** (yolov8n-pose.pt): typically person.
   - Detection.source="pose".

4. **Perception (Geometry: geom_circle_detector)**
   - Outputs Dict[frame_idx, List[Detection]].
   - Detects circles using HoughCircles + verification and emits class_name="discos" with:
     - Detection.source="geom"
     - Detection.radius_px (circle radius)
     - Detection.shape_score (verification score).

5. **Merge (detection_merger)**
   - Concatenates detections from multiple sources per frame.
   - Important: The merge is *append-only*; it does not deduplicate.

6. **Tracking (byte_tracker)**
   - Converts detections into TrackedObjects with IDs.
   - Critical invariant: A track ID must not "change species".
   - Because different models can reuse class_id values (e.g. Pose person is often 0, custom atleta can also be 0), matching must be based on class_name not class_id.

7. **Hybrid Calibration (disc_calibrator)**
   - Looks at the first **N frames** (configurable) and matches YOLO-disc with geom-disc.
   - Produces a stable artifact disc_prior.json with seed_center/seed_radius/sigmas.

8. **Hybrid Tracking (disc_fusion_tracker)**
   - Uses Kalman prediction and fuses measurements from YOLO+geom near the predicted position.
   - Outputs a unified discos master track and writes rich debug artifacts.

9. **Visualization (video_renderer)**
   - Draws masks/boxes and keypoints when present.

## 2. Typical Failure Modes

## 2.1 Pose appears only in some frames (athlete "blinks")
Common causes:
- Pose conf_threshold too high.
- Fast motion blur.

Mitigation:
- Lower pose_estimation.params.conf_threshold.
- Forward-fill athlete bbox for ROI gating, but **never forward-fill keypoints** (otherwise the skeleton "freezes").

## 2.2 Disc is tracked at the start, then drops when bar moves
Common causes:
- ROI too tight (disc leaves the athlete bbox band when the bar rises).
- Search radius too small (Kalman prediction + motion exceeds gating).
- Shape/size gates too strict once the disc bbox deforms during motion blur.

Mitigation:
- Use per-frame telemetry (JSONL) to see which gate rejects candidates: ROI, radius, ratio, area_ratio, cost.
- Tune ratio_min_tracking, area_ratio_min_tracking, LOST multipliers, and ROI margins.

## 3. Debug Telemetry

The hybrid tracker writes debug artifacts into the run output directory (same folder as pipeline.log):
- disc_prior.json: calibration output (stable name).
- disc_tracker_debug.jsonl: per-frame evaluation log.
- disc_tracker_debug_summary.json: run summary (first lost frame, rejection reason counts, #frames tracked).

Each line is a JSON object for one frame, containing:
- frame: frame index
- state: TRACKING/LOST
- pred: predicted center from Kalman
- roi: ROI bounds used
- dist_gate / rad_gate: dynamic gates derived from the calibrated sigmas
- candidates_total
- best: chosen candidate metrics if accepted
- candidates: top-K candidates including rejection reasons (roi, dist, radius) and computed costs
- accepted: boolean

This file is the primary tool to answer: "why did we lose the disc exactly at frame N?"
