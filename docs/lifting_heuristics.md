# Lifting Heuristics (Business Logic)

This document describes the heuristics used for tracking objects of interest in a weightlifting video (side view).

## Current Architecture: Three-Phase Heuristics

Heuristics are separated into three phases based on what information they need:

### Phase 1: Pre-Tracking (DetectionFilter)
- **No temporal state** - operates on individual frames
- **Size Filter**: Disc must be within reference radius ± tolerance (e.g., 30%)
- **Largest Selector**: Only the largest athlete per frame

### Phase 2: Tracking (ModelTracker)
- **Temporal state** (Kalman filter)
- **Single-Object Mode**: Only one disc tracked
- **Hungarian Assignment**: Optimal matching to predictions

### Phase 3: Post-Tracking (TrackRefiner)
- **Full trajectory** available
- **Smoothing**: Moving average filter
- **Outlier Removal**: (Future) Remove sudden jumps

---

## 1. The Single Disc Rule

We track only ONE disc of interest throughout the video.

**Implementation**:
1. **Manual Selection**: Use `select_disc.py` to select center/radius in frame 0
2. **Size Filter**: Only consider detections within radius ± tolerance
3. **Single-Object Tracker**: Maintain only one track, initialized from selection

**Config**:
```yaml
size_filter:
  enabled: true
  selection_file: "../data/outputs/disc_selection.json"
  tolerance: 0.30
  classes: ["discos"]

initial_selection:
  class_name: "discos"
  selection_file: "../data/outputs/disc_selection.json"
```

---

## 2. The Single Athlete Rule

We track only ONE athlete of interest - typically the largest in frame.

**Heuristic**: Keep only the detection with largest bounding box area per frame.

**Rationale**:
- Camera is focused on the athlete
- The athlete of interest is usually the largest person in frame
- Occasionally someone may cross in front, but this is handled by the "largest" heuristic

**Config**:
```yaml
largest_selector:
  enabled: true
  classes: ["atleta"]  # or "person" for COCO
```

**Known Limitations**:
- If someone walks very close to the camera, they may temporarily be larger
- Future improvement: Track consistency across frames

---

## 3. Size Consistency Rule

The disc doesn't change size significantly during the lift.

**Implementation**:
- Reference radius is set from manual selection
- Size tolerance (default 30%) allows for:
  - Motion blur
  - Partial occlusions
  - Detection quality variations
  - Camera perspective changes

---

## 4. Trajectory Smoothing

Raw detections can be noisy. We apply post-tracking smoothing.

**Methods**:
- **Moving Average**: Simple, robust to outliers
- **Exponential**: More weight on recent values
- **Savitzky-Golay**: Preserves peaks and edges (requires scipy)

**Config**:
```yaml
smoothing:
  enabled: true
  method: "moving_average"
  window: 5
```

---

## 5. Future Heuristics (Not Yet Implemented)

### Direction Constraints
- The disc should move mostly vertically during the lift
- Excessive horizontal movement may indicate tracking error

### Appearance/Disappearance Rules
- The disc of interest doesn't disappear (it's always visible)
- Can be used to validate track continuity

### Wrist Proximity
- Disc should be near the athlete's wrists during the lift
- Helps distinguish active disc from floor discs
