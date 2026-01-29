"""
ModelTracker: Independent per-model tracking with Kalman filter and Hungarian assignment.

Features:
- Kalman filter (constant velocity model) for state prediction
- Track lifecycle: TENTATIVE → CONFIRMED → LOST → TERMINATED
- Configurable cost function: IoU + center distance + scale
- Dual threshold support (high/low confidence)
- Per-class separation (no cross-class matching)
- Single-object mode with initial selection (for disc tracking)
- Outputs: tracked_detections.jsonl, tracks_summary.json

Based on tracking-by-detection paradigm with robust defaults.

Note: Size filtering should be handled by DetectionFilter (pre-tracking step).
This module focuses purely on temporal association (tracking).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import linear_sum_assignment

from domain.ports import IPipelineStep
from domain.entities import Detection, TrackedObject


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = "tentative"    # New track, not yet confirmed
    CONFIRMED = "confirmed"    # Stable track with enough hits
    LOST = "lost"              # Missing detections, still searchable
    TERMINATED = "terminated"  # Dead track, will be removed


@dataclass
class KalmanState:
    """
    2D Kalman filter state for position + velocity.
    State: [cx, cy, vx, vy]
    Measurement: [cx, cy]
    """
    # State vector [cx, cy, vx, vy]
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # State covariance matrix
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 100.0)
    
    # Process noise (how much we trust the model)
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 5.0, 5.0]))
    # Measurement noise (how much we trust detections)
    R: np.ndarray = field(default_factory=lambda: np.diag([5.0, 5.0]))
    
    # State transition matrix (constant velocity)
    F: np.ndarray = field(default_factory=lambda: np.array([
        [1, 0, 1, 0],  # cx' = cx + vx
        [0, 1, 0, 1],  # cy' = cy + vy
        [0, 0, 1, 0],  # vx' = vx
        [0, 0, 0, 1],  # vy' = vy
    ], dtype=float))
    
    # Measurement matrix
    H: np.ndarray = field(default_factory=lambda: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float))
    
    def predict(self) -> Tuple[float, float]:
        """Predict next state. Returns predicted (cx, cy)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])
    
    def update(self, cx: float, cy: float) -> None:
        """Update state with measurement."""
        z = np.array([cx, cy])
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])
    
    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


@dataclass
class Track:
    """A single tracked object."""
    track_id: int
    class_name: str
    state: TrackState
    kalman: KalmanState
    
    # Lifecycle counters
    hits: int = 0              # Total frames with match
    missed_frames: int = 0     # Consecutive frames without match
    age_frames: int = 0        # Total frames since creation
    
    # Last detection info
    last_bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    last_score: float = 0.0
    last_frame_idx: int = 0
    
    # Reference size (for size-constrained tracking)
    reference_radius: Optional[float] = None
    
    # Trajectory history (for visualization)
    trajectory: List[Tuple[int, float, float]] = field(default_factory=list)
    
    def to_snapshot(self) -> Dict[str, Any]:
        """Export track state for debugging/output."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "state": self.state.value,
            "hits": self.hits,
            "missed_frames": self.missed_frames,
            "age_frames": self.age_frames,
            "last_frame_idx": self.last_frame_idx,
            "last_bbox": list(self.last_bbox),
            "last_score": self.last_score,
            "position": list(self.kalman.position),
            "velocity": list(self.kalman.velocity),
            "reference_radius": self.reference_radius,
        }


def box_iou(box1: Tuple, box2: Tuple) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou


def box_center(box: Tuple) -> Tuple[float, float]:
    """Get center of box [x1, y1, x2, y2]."""
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def box_radius(box: Tuple) -> float:
    """Get approximate radius of box (half of average dimension)."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (w + h) / 4.0  # Average of w/2 and h/2


def center_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


class TrackIdGenerator:
    """Generates unique track IDs across all trackers."""
    
    def __init__(self, start: int = 1):
        self._next = start
    
    def next(self) -> int:
        """Get next unique track ID."""
        tid = self._next
        self._next += 1
        return tid


class SingleObjectTracker:
    """
    Specialized tracker for a SINGLE object with known initial position.
    
    Used when initial_selection is provided. Key differences from ClassTracker:
    - Only ONE track is maintained (no new tracks created)
    - Track is pre-initialized from selection (starts as CONFIRMED)
    - Track never terminates (object doesn't disappear by assumption)
    - Best candidate selection based on distance to Kalman prediction
    
    Note: Size filtering should be done by DetectionFilter BEFORE this tracker.
    This keeps the tracker focused on temporal association only.
    """
    
    def __init__(
        self, 
        class_name: str, 
        config: Dict[str, Any], 
        selection: Dict[str, Any],
        id_generator: Optional[TrackIdGenerator] = None
    ):
        self.class_name = class_name
        self.config = config
        self.selection = selection
        self.id_generator = id_generator or TrackIdGenerator()
        
        # Extract selection data
        center = selection.get("center")
        radius = selection.get("radius", 50.0)  # Default radius for bbox estimation
        
        if center is None:
            raise ValueError(f"SingleObjectTracker requires center in selection")
        
        self.reference_center = tuple(center) if isinstance(center, list) else center
        self.reference_radius = float(radius) if radius else 50.0
        
        # Config
        self.min_det_score = config.get("min_det_score", 0.05)
        self.max_age_frames = config.get("max_age_frames", 30)  # More tolerance for single object
        
        # Association config
        assoc = config.get("association", {})
        self.max_center_dist_px = assoc.get("max_center_dist_px", 200)
        
        # Initialize the single track
        self.track = self._create_initial_track()
        self.terminated = False
        
        print(f"[SingleObjectTracker] Initialized for '{class_name}'")
        print(f"[SingleObjectTracker] Reference center: {self.reference_center}")
        print(f"[SingleObjectTracker] Max association distance: {self.max_center_dist_px}px")
    
    def _create_initial_track(self) -> Track:
        """Create the initial track from selection."""
        cx, cy = self.reference_center
        r = self.reference_radius
        
        # Create bbox from circle
        bbox = (cx - r, cy - r, cx + r, cy + r)
        
        kalman = KalmanState()
        kalman.x = np.array([cx, cy, 0.0, 0.0])
        kalman.P = np.eye(4) * 10.0  # Low initial uncertainty (we trust the selection)
        
        track = Track(
            track_id=self.id_generator.next(),  # Unique ID from shared generator
            class_name=self.class_name,
            state=TrackState.CONFIRMED,  # Start as confirmed (trusted selection)
            kalman=kalman,
            hits=1,
            missed_frames=0,
            age_frames=0,
            last_bbox=bbox,
            last_score=1.0,  # Perfect confidence for manual selection
            last_frame_idx=-1,
            reference_radius=self.reference_radius,
            trajectory=[(0, cx, cy)],
        )
        return track
    
    def update(self, frame_idx: int, detections: List[Detection]) -> List[Tuple[Track, Detection, str]]:
        """
        Update the single track with detections.
        
        Assumes detections have already been filtered by DetectionFilter.
        This method only does temporal association (find closest to prediction).
        
        Returns list with at most one (track, detection, status) tuple.
        """
        results = []
        
        if self.terminated:
            return results
        
        # Predict
        pred_cx, pred_cy = self.track.kalman.predict()
        self.track.age_frames += 1
        
        # Find best candidate (closest to prediction within distance gate)
        valid_candidates = []
        
        for det in detections:
            # Basic score filter (very low threshold, main filtering done pre-tracking)
            if det.confidence < self.min_det_score:
                continue
            
            # Distance to prediction
            det_cx, det_cy = box_center(det.bbox)
            dist = center_distance((pred_cx, pred_cy), (det_cx, det_cy))
            
            if dist <= self.max_center_dist_px:
                valid_candidates.append((det, dist))
        
        # Select best candidate (closest to prediction)
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[1])
            best_det, best_dist = valid_candidates[0]
            
            # Update track
            cx, cy = box_center(best_det.bbox)
            self.track.kalman.update(cx, cy)
            self.track.last_bbox = best_det.bbox
            self.track.last_score = best_det.confidence
            self.track.last_frame_idx = frame_idx
            self.track.hits += 1
            self.track.missed_frames = 0
            self.track.trajectory.append((frame_idx, cx, cy))
            
            if self.track.state == TrackState.LOST:
                self.track.state = TrackState.CONFIRMED
            
            results.append((self.track, best_det, "matched"))
        else:
            # No valid candidate
            self.track.missed_frames += 1
            
            if self.track.state == TrackState.CONFIRMED:
                self.track.state = TrackState.LOST
            
            # Note: We don't terminate single object tracks (heuristic: object doesn't disappear)
        
        return results
    
    def get_all_tracks(self) -> List[Track]:
        """Return the single track."""
        return [self.track]


class ClassTracker:
    """
    Tracker for a single class. Maintains tracks and performs association.
    Separation by class avoids absurd switches (e.g., disc ↔ person).
    """
    
    def __init__(self, class_name: str, config: Dict[str, Any], id_generator: Optional[TrackIdGenerator] = None):
        self.class_name = class_name
        self.config = config
        self.tracks: List[Track] = []
        self.id_generator = id_generator or TrackIdGenerator()
        self.terminated_tracks: List[Track] = []
        
        # Config with defaults
        self.min_det_score = config.get("min_det_score", 0.05)
        self.high_det_score = config.get("high_det_score", 0.25)
        self.max_age_frames = config.get("max_age_frames", 15)
        self.min_hits_to_confirm = config.get("min_hits_to_confirm", 2)
        
        # Association config
        assoc = config.get("association", {})
        self.iou_weight = assoc.get("iou_weight", 0.5)
        self.center_weight = assoc.get("center_weight", 0.5)
        self.max_center_dist_px = assoc.get("max_center_dist_px", 150)
        self.min_iou = assoc.get("min_iou", 0.01)
    
    def _compute_cost(self, track: Track, det: Detection, pred_cx: float, pred_cy: float) -> float:
        """
        Compute association cost between track and detection.
        Lower is better. Returns high cost if gating fails.
        """
        det_cx, det_cy = box_center(det.bbox)
        
        # Center distance (normalized)
        dist = center_distance((pred_cx, pred_cy), (det_cx, det_cy))
        if dist > self.max_center_dist_px:
            return 1000.0  # Gating: too far
        norm_dist = dist / self.max_center_dist_px
        
        # IoU with predicted bbox
        # Estimate predicted bbox from track's last bbox shifted by velocity
        vx, vy = track.kalman.velocity
        w = track.last_bbox[2] - track.last_bbox[0]
        h = track.last_bbox[3] - track.last_bbox[1]
        pred_bbox = (pred_cx - w/2, pred_cy - h/2, pred_cx + w/2, pred_cy + h/2)
        
        iou = box_iou(pred_bbox, det.bbox)
        if iou < self.min_iou:
            return 1000.0  # Gating: no overlap
        
        # Combined cost
        cost = self.center_weight * norm_dist + self.iou_weight * (1.0 - iou)
        return cost
    
    def update(self, frame_idx: int, detections: List[Detection]) -> List[Tuple[Track, Detection, str]]:
        """
        Update tracks with new detections for this class.
        Returns list of (track, detection, status) tuples.
        Status: "matched", "new_track", "unmatched_track"
        """
        results = []
        
        # Filter detections by score
        valid_dets = [d for d in detections if d.confidence >= self.min_det_score]
        high_dets = [d for d in valid_dets if d.confidence >= self.high_det_score]
        low_dets = [d for d in valid_dets if d.confidence < self.high_det_score]
        
        # Get active tracks (not terminated)
        active_tracks = [t for t in self.tracks if t.state != TrackState.TERMINATED]
        
        # Predict all tracks
        predictions = {}
        for track in active_tracks:
            pred_cx, pred_cy = track.kalman.predict()
            predictions[track.track_id] = (pred_cx, pred_cy)
            track.age_frames += 1
        
        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()
        
        # --- First pass: Match high-confidence detections ---
        if active_tracks and high_dets:
            cost_matrix = np.zeros((len(active_tracks), len(high_dets)))
            for i, track in enumerate(active_tracks):
                pred_cx, pred_cy = predictions[track.track_id]
                for j, det in enumerate(high_dets):
                    cost_matrix[i, j] = self._compute_cost(track, det, pred_cx, pred_cy)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 100.0:  # Valid match
                    track = active_tracks[r]
                    det = high_dets[c]
                    self._update_track(track, det, frame_idx)
                    matched_tracks.add(track.track_id)
                    matched_dets.add(id(det))
                    results.append((track, det, "matched"))
        
        # --- Second pass: Match low-confidence detections to unmatched CONFIRMED tracks ---
        # (ByteTrack-like: use low detections to avoid cutting tracks)
        unmatched_tracks = [t for t in active_tracks 
                          if t.track_id not in matched_tracks 
                          and t.state == TrackState.CONFIRMED]
        unmatched_low_dets = [d for d in low_dets if id(d) not in matched_dets]
        
        if unmatched_tracks and unmatched_low_dets:
            cost_matrix = np.zeros((len(unmatched_tracks), len(unmatched_low_dets)))
            for i, track in enumerate(unmatched_tracks):
                pred_cx, pred_cy = predictions[track.track_id]
                for j, det in enumerate(unmatched_low_dets):
                    cost_matrix[i, j] = self._compute_cost(track, det, pred_cx, pred_cy)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 100.0:
                    track = unmatched_tracks[r]
                    det = unmatched_low_dets[c]
                    self._update_track(track, det, frame_idx)
                    matched_tracks.add(track.track_id)
                    matched_dets.add(id(det))
                    results.append((track, det, "matched"))
        
        # --- Handle unmatched tracks ---
        for track in active_tracks:
            if track.track_id not in matched_tracks:
                track.missed_frames += 1
                
                if track.missed_frames > self.max_age_frames:
                    track.state = TrackState.TERMINATED
                    self.terminated_tracks.append(track)
                elif track.state == TrackState.CONFIRMED:
                    track.state = TrackState.LOST
                elif track.state == TrackState.TENTATIVE:
                    # Tentative tracks die faster
                    if track.missed_frames > 2:
                        track.state = TrackState.TERMINATED
                        self.terminated_tracks.append(track)
        
        # Remove terminated tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.TERMINATED]
        
        # --- Create new tracks from unmatched high-confidence detections ---
        for det in high_dets:
            if id(det) not in matched_dets:
                new_track = self._create_track(det, frame_idx)
                self.tracks.append(new_track)
                results.append((new_track, det, "new_track"))
        
        return results
    
    def _update_track(self, track: Track, det: Detection, frame_idx: int) -> None:
        """Update track with matched detection."""
        cx, cy = box_center(det.bbox)
        track.kalman.update(cx, cy)
        track.last_bbox = det.bbox
        track.last_score = det.confidence
        track.last_frame_idx = frame_idx
        track.hits += 1
        track.missed_frames = 0
        track.trajectory.append((frame_idx, cx, cy))
        
        # State transitions
        if track.state == TrackState.TENTATIVE:
            if track.hits >= self.min_hits_to_confirm:
                track.state = TrackState.CONFIRMED
        elif track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED
    
    def _create_track(self, det: Detection, frame_idx: int) -> Track:
        """Create a new track from a detection."""
        cx, cy = box_center(det.bbox)
        
        kalman = KalmanState()
        kalman.x = np.array([cx, cy, 0.0, 0.0])
        
        track = Track(
            track_id=self.id_generator.next(),
            class_name=self.class_name,
            state=TrackState.TENTATIVE,
            kalman=kalman,
            hits=1,
            missed_frames=0,
            age_frames=1,
            last_bbox=det.bbox,
            last_score=det.confidence,
            last_frame_idx=frame_idx,
            trajectory=[(frame_idx, cx, cy)],
        )
        return track
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks including terminated."""
        return self.tracks + self.terminated_tracks


class ModelTracker(IPipelineStep[Dict[int, List[Detection]], Dict[int, List[TrackedObject]]]):
    """
    Per-model object tracker.
    
    Applies tracking-by-detection with Kalman filter and Hungarian assignment.
    Tracks are separated by class to avoid cross-class ID switches.
    
    Config:
        enabled: bool (default True)
        classes_to_track: List[str] or None (track all if None)
        min_det_score: float (default 0.05)
        high_det_score: float (default 0.25)
        max_age_frames: int (default 15)
        min_hits_to_confirm: int (default 2)
        
        # Initial selection for single-object tracking
        # Note: Size filtering should be done by DetectionFilter
        initial_selection:
            class_name: str (e.g., "discos")
            center: [x, y]  # Required
            radius: float   # Optional (for bbox estimation)
            selection_file: str (optional, load from file)
        
        association:
            iou_weight: float (default 0.5)
            center_weight: float (default 0.5)
            max_center_dist_px: float (default 150)
            min_iou: float (default 0.01)
        output:
            save_tracked_detections: bool (default True)
            save_tracks_summary: bool (default True)
    """
    
    def __init__(self):
        self.class_trackers: Dict[str, ClassTracker] = {}
        self.single_trackers: Dict[str, SingleObjectTracker] = {}
        self.config: Dict[str, Any] = {}
        self.frame_results: List[Dict] = []  # For JSONL output
        self.id_generator = TrackIdGenerator()  # Shared across all trackers
    
    def run(self, input_data: Dict[int, List[Detection]], config: Dict[str, Any]) -> Dict[int, List[TrackedObject]]:
        """
        Run tracking on detections.
        
        Args:
            input_data: Dict[frame_idx, List[Detection]]
            config: Tracking configuration
            
        Returns:
            Dict[frame_idx, List[TrackedObject]]
        """
        self.config = config
        self.class_trackers = {}
        self.single_trackers = {}
        self.frame_results = []
        
        # Check if tracking is enabled
        if not config.get("enabled", True):
            # Pass through without tracking
            return self._passthrough(input_data)
        
        classes_to_track = config.get("classes_to_track", None)
        progress_every = config.get("progress_every", 50)
        
        # Check for initial selection (single-object mode)
        initial_selection = config.get("initial_selection")
        if initial_selection:
            # Load from file if path provided
            # Check multiple sources: config, or _selection_file injected by pipeline
            selection_file = initial_selection.get("selection_file") or config.get("_selection_file")
            if selection_file:
                selection_path = Path(selection_file)
                if not selection_path.is_absolute():
                    # Relative to project root / ai-core
                    project_root = Path(config.get("_project_root", "."))
                    selection_path = project_root / "ai-core" / selection_file
                    if not selection_path.exists():
                        selection_path = project_root / selection_file
                
                if selection_path.exists():
                    with open(selection_path, 'r') as f:
                        file_data = json.load(f)
                    # Merge file data with config (file takes precedence for center/radius)
                    if file_data.get("center"):
                        initial_selection["center"] = file_data["center"]
                    if file_data.get("radius"):
                        initial_selection["radius"] = file_data["radius"]
                    print(f"[ModelTracker] Loaded selection from: {selection_path}")
                else:
                    print(f"[ModelTracker] Warning: Selection file not found: {selection_path}")
            
            selection_class = initial_selection.get("class_name")
            if selection_class:
                print(f"[ModelTracker] Single-object mode enabled for class '{selection_class}'")
                self.single_trackers[selection_class] = SingleObjectTracker(
                    selection_class, config, initial_selection, self.id_generator
                )
        
        frame_indices = sorted(input_data.keys())
        if not frame_indices:
            return {}
        
        total_frames = len(frame_indices)
        print(f"[ModelTracker] Starting tracking for {total_frames} frames...")
        
        results: Dict[int, List[TrackedObject]] = {}
        
        for i, frame_idx in enumerate(frame_indices):
            detections = input_data[frame_idx]
            
            # Filter by classes if specified
            if classes_to_track:
                detections = [d for d in detections if d.class_name in classes_to_track]
            
            # Group detections by class
            dets_by_class: Dict[str, List[Detection]] = {}
            for det in detections:
                if det.class_name not in dets_by_class:
                    dets_by_class[det.class_name] = []
                dets_by_class[det.class_name].append(det)
            
            frame_tracked = []
            frame_debug = {"frame_idx": frame_idx, "tracks": []}
            
            # Update each class tracker
            for class_name, class_dets in dets_by_class.items():
                # Check if this class uses single-object tracker
                if class_name in self.single_trackers:
                    tracker = self.single_trackers[class_name]
                    updates = tracker.update(frame_idx, class_dets)
                else:
                    # Standard multi-object tracker
                    if class_name not in self.class_trackers:
                        self.class_trackers[class_name] = ClassTracker(class_name, config, self.id_generator)
                    
                    tracker = self.class_trackers[class_name]
                    updates = tracker.update(frame_idx, class_dets)
                
                # Convert to TrackedObject
                for track, det, status in updates:
                    if track.state in (TrackState.CONFIRMED, TrackState.TENTATIVE):
                        tracked_obj = TrackedObject(
                            track_id=track.track_id,
                            detection=det,
                            history=[(cx, cy) for _, cx, cy in track.trajectory[-30:]],
                            velocity=track.kalman.velocity,
                        )
                        frame_tracked.append(tracked_obj)
                        
                        frame_debug["tracks"].append({
                            "track_id": track.track_id,
                            "class_name": track.class_name,
                            "state": track.state.value,
                            "status": status,
                            "bbox": list(det.bbox),
                            "score": det.confidence,
                            "reference_radius": track.reference_radius,
                        })
            
            # For single-object trackers, also emit even when LOST (for debugging)
            for class_name, tracker in self.single_trackers.items():
                if tracker.track.state == TrackState.LOST and class_name not in dets_by_class:
                    frame_debug["tracks"].append({
                        "track_id": tracker.track.track_id,
                        "class_name": class_name,
                        "state": "lost",
                        "status": "no_detection",
                        "missed_frames": tracker.track.missed_frames,
                    })
            
            results[frame_idx] = frame_tracked
            self.frame_results.append(frame_debug)
            
            # Progress indicator
            if progress_every > 0 and (i + 1) % progress_every == 0:
                pct = (i + 1) / total_frames * 100
                print(f"[ModelTracker] Progress: {i+1}/{total_frames} ({pct:.1f}%)")
        
        # Final stats
        all_trackers = list(self.class_trackers.values()) + list(self.single_trackers.values())
        total_tracks = sum(len(t.get_all_tracks()) for t in all_trackers)
        
        # Count active tracks
        active_count = 0
        for t in self.class_trackers.values():
            active_count += len([tr for tr in t.tracks if tr.state == TrackState.CONFIRMED])
        for t in self.single_trackers.values():
            if t.track.state in (TrackState.CONFIRMED, TrackState.LOST):
                active_count += 1
        
        print(f"[ModelTracker] Complete. Total tracks: {total_tracks}, Active at end: {active_count}")
        
        return results
    
    def _passthrough(self, input_data: Dict[int, List[Detection]]) -> Dict[int, List[TrackedObject]]:
        """Pass through detections without tracking (assign sequential IDs)."""
        results = {}
        next_id = 1
        for frame_idx, dets in input_data.items():
            frame_tracked = []
            for det in dets:
                tracked_obj = TrackedObject(
                    track_id=next_id,
                    detection=det,
                    history=[box_center(det.bbox)],
                    velocity=None,
                )
                frame_tracked.append(tracked_obj)
                next_id += 1
            results[frame_idx] = frame_tracked
        return results
    
    def save_result(self, data: Dict[int, List[TrackedObject]], output_path: Path) -> None:
        """Save tracked detections and summary."""
        output_config = self.config.get("output", {})
        output_dir = output_path.parent
        base_name = output_path.stem
        
        # Main output (standard format)
        serializable = {
            str(k): [t.model_dump() for t in v]
            for k, v in data.items()
        }
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        # JSONL per-frame debug
        if output_config.get("save_tracked_detections", True):
            jsonl_path = output_dir / f"{base_name}_debug.jsonl"
            with open(jsonl_path, 'w') as f:
                for frame_data in self.frame_results:
                    f.write(json.dumps(frame_data) + "\n")
        
        # Summary
        if output_config.get("save_tracks_summary", True):
            summary = self._generate_summary()
            summary_path = output_dir / f"{base_name}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate tracks summary for output."""
        all_tracks = []
        
        # Standard trackers
        for class_name, tracker in self.class_trackers.items():
            for track in tracker.get_all_tracks():
                all_tracks.append({
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "state": track.state.value,
                    "hits": track.hits,
                    "age_frames": track.age_frames,
                    "trajectory_length": len(track.trajectory),
                    "first_frame": track.trajectory[0][0] if track.trajectory else None,
                    "last_frame": track.trajectory[-1][0] if track.trajectory else None,
                    "avg_score": track.last_score,
                    "mode": "multi_object",
                })
        
        # Single-object trackers
        for class_name, tracker in self.single_trackers.items():
            track = tracker.track
            all_tracks.append({
                "track_id": track.track_id,
                "class_name": track.class_name,
                "state": track.state.value,
                "hits": track.hits,
                "age_frames": track.age_frames,
                "trajectory_length": len(track.trajectory),
                "first_frame": track.trajectory[0][0] if track.trajectory else None,
                "last_frame": track.trajectory[-1][0] if track.trajectory else None,
                "avg_score": track.last_score,
                "mode": "single_object",
                "reference_center": tracker.reference_center,
            })
        
        return {
            "total_tracks": len(all_tracks),
            "tracks_by_class": {
                cls: len([t for t in all_tracks if t["class_name"] == cls])
                for cls in set(t["class_name"] for t in all_tracks)
            },
            "single_object_classes": list(self.single_trackers.keys()),
            "tracks": all_tracks,
        }
    
    def load_result(self, input_path: Path) -> Dict[int, List[TrackedObject]]:
        """Load tracked detections from file."""
        with open(input_path, 'r') as f:
            raw = json.load(f)
        return {
            int(k): [TrackedObject(**t) for t in v]
            for k, v in raw.items()
        }
