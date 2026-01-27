# Lifting Heuristics (Business Logic)

This document describes the biomechanical and morphological rules used by the LiftingSessionOptimizer to robustly track the barbell plate in a weightlifting video (side view).

## 0. Hybrid Strategy (Recommended)
In practice, the segmentation model (best.pt) can temporarily fail when the bar starts moving (motion blur, occlusions, class flipping).
To keep the system robust and debuggable, we support a **hybrid approach**:

- **YOLO segmentation** provides rich semantics and masks when it works well.
- **Geometric disc detection** (HoughCircles + verification) provides a complementary "shape sensor".
- A short **calibration window** (default 30 frames, configurable) uses agreement between both to establish a stable prior:
  - seed_center, seed_radius, and uncertainty sigmas.
- A **fusion tracker** (Kalman + gating) fuses YOLO and geometric candidates near the predicted position.

The key outcome is still a single unified domain label: class_name="discos"; provenance is tracked via Detection.source.

### Wrist proximity (optional, recommended)
To avoid selecting discs resting on the floor, we can add a **wrist-distance penalty** based on pose keypoints:
- Compute the midpoint of visible wrists (COCO keypoints 9 and 10).
- Add a configurable penalty proportional to disc distance from wrists.
- With a high weight, this behaves almost like a hard gate.

This penalty should ideally only be applied when pose quality is sufficient (future improvement).

## 1. The Single Athlete Rule (Primary Actor)
In a valid analysis video, there should be only **one** primary athlete performing the lift.
*   **Heuristic:** Distance to Center of Frame.
*   **Logic:**
    1.  Calculate the centroid of every "person" track (from Pose Model) throughout the video duration.
    2.  Select the track with the **minimum average distance** to the video center.
    3.  Discard all other "person" tracks.

## 2. The Squareness Rule (Morphology)
We leverage the specific geometry of the recording setup: **Side View**.
*   **Heuristic:** Bounding Box Aspect Ratio.
*   **Logic:**
    *   A circular weight plate viewed from the side (orthogonal to the camera axis) appears as a circle.
    *   Therefore, its 2D Bounding Box must be approximately **Square**.
    *   **Filter:** min(w, h) / max(w, h) > Threshold (e.g., 0.6).

## 3. Size Consistency Rule
A physical plate does not change size significantly during the lift.
*   **Heuristic:** Area Similarity.
*   **Logic:**
    *   Once a candidate plate is identified, valid future detections must have a similar area (e.g., within 20% tolerance).

## 4. Custom Morphological Tracking (The "Master Track")
We reconstruct the "Master Track" of the plate using a robust predictive algorithm.

*   **Step 1: Seed Selection.**
    *   Look at the first N frames.
    *   Find the object that is **Largest** (Area), **Most Square** (Ratio ~ 1.0), and located in the **Lower Half** of the image.

*   **Step 2: Trajectory Tracing with Linear Prediction.**
    *   **Prediction:** Instead of assuming the plate is static, we calculate its velocity vector (vx, vy) based on the last few frames.
    *   **Search Area:** We center our search for the next candidate at Predicted_Pos = Current_Pos + Velocity.
    *   **Score:** Proximity to predicted position, Area Similarity, and Squareness.

*   **Step 3: Dynamic Tolerance Relaxation.**
    *   If no candidate is found with strict parameters (e.g., due to motion blur deforming the shape):
    *   **Retry 1:** Relax area tolerance (e.g., 20% -> 30%) and search radius (1.5x).
    *   **Retry 2:** Relax further (2.0x).
    *   This ensures we track the plate even when detection quality degrades during fast movement.

## 5. Geometric Disc Detection (HoughCircles + Verification)
When available, geometric detections are produced by:
- HoughCircles on tiled image regions for proposals
- Verification with 3 key metrics: contrast_frac, arc_frac, interior_edge_density

We keep only circle candidates that pass verification, and compute:
- **Center**: circle center (cx, cy)
- **Radius**: circle radius r
- **shape_score**: composite verification score

These detections are emitted as class_name="discos" with source="geom".

## 6. Calibration by Consensus (First N Frames)
In the first calibration_frames frames we try to match:
- best YOLO-disc and best geom-disc for that frame

If they agree (distance + radius similarity), we aggregate across frames and compute:
- seed_center: median center
- seed_radius: median radius
- sigma_center / sigma_radius: MAD-based uncertainty

This becomes our prior (disc_prior.json) for the fusion tracker.
