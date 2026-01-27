import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from scipy.optimize import linear_sum_assignment

from domain.ports import IPipelineStep
from domain.entities import Detection, TrackedObject

# Basic Intersection over Union (IoU)
def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

class ByteTracker(IPipelineStep[Dict[int, List[Detection]], Dict[int, List[TrackedObject]]]):
    """
    A simplified implementation of ByteTrack logic for object tracking.
    Matches detections to existing tracks based on IoU.
    """

    def __init__(self):
        self.tracks: List[TrackedObject] = []
        self.next_id = 1
        # Configurable params
        self.match_thresh = 0.8 # High confidence match
        self.track_buffer = 30 # Frames to keep lost tracks

    def run(self, input_data: Dict[int, List[Detection]], config: Dict[str, Any]) -> Dict[int, List[TrackedObject]]:
        """
        Runs tracking on the detections.
        Input: Frame -> List[Detection]
        Output: Frame -> List[TrackedObject]
        """
        results: Dict[int, List[TrackedObject]] = {}
        
        # Reset state for new run
        self.tracks = []
        self.next_id = 1
        
        frame_indices = sorted(input_data.keys())
        if not frame_indices:
            return {}
            
        print(f"Tracking objects across {len(frame_indices)} frames...")
        
        # We need to maintain state across frames.
        # Active tracks: Currently matched
        # Lost tracks: Matched recently but not now
        # Removed tracks: Gone
        
        # Simplified State: Just a list of active tracks with 'last_seen_frame'
        active_tracks: List[Dict] = [] # {'id': int, 'bbox': [], 'history': [], 'last_seen': int, 'velocity': (vx, vy)}
        
        for frame_idx in frame_indices:
            detections = input_data[frame_idx]
            frame_tracks = []
            
            # 1. Prediction (Kalman Filter would go here, for now use Linear velocity)
            for t in active_tracks:
                # Predict next position based on velocity
                if t.get('velocity'):
                    vx, vy = t['velocity']
                    w = t['bbox'][2] - t['bbox'][0]
                    h = t['bbox'][3] - t['bbox'][1]
                    cx = t['bbox'][0] + w/2 + vx
                    cy = t['bbox'][1] + h/2 + vy
                    t['predicted_bbox'] = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                else:
                    t['predicted_bbox'] = t['bbox']

            # 2. Matching
            # Compute Cost Matrix (1 - IoU)
            # IMPORTANT: We should NOT match objects of different classes!
            # ByteTrack original implementation matches everything, but for us,
            # an 'atleta' should never become a 'disco'.
            
            if active_tracks and detections:
                cost_matrix = np.zeros((len(active_tracks), len(detections)))
                for i, t in enumerate(active_tracks):
                    for j, d in enumerate(detections):
                        # Strict Class Check
                        # IMPORTANT: When merging outputs from multiple models, class_id may collide
                        # (e.g. Pose model 'person' can be class_id=0 while custom model 'atleta' is also 0).
                        # Therefore, class integrity must be enforced by class_name (semantic label), not class_id.
                        if t['detection'].class_name != d.class_name:
                            cost_matrix[i, j] = 1000.0 # Impossible match
                        else:
                            # Use predicted bbox for matching
                            iou = box_iou(t['predicted_bbox'], d.bbox)
                            cost_matrix[i, j] = 1.0 - iou
                
                # Hungrian Algorithm (Linear Sum Assignment)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                matches = []
                unmatched_tracks = set(range(len(active_tracks)))
                unmatched_detections = set(range(len(detections)))
                
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < (1.0 - 0.1): # IoU Threshold 0.1
                        matches.append((r, c))
                        unmatched_tracks.discard(r)
                        unmatched_detections.discard(c)
                    else:
                        pass # Too far
                
                # Update Matched Tracks
                for r, c in matches:
                    track = active_tracks[r]
                    det = detections[c]
                    
                    # Update Velocity
                    prev_cx = (track['bbox'][0] + track['bbox'][2]) / 2
                    prev_cy = (track['bbox'][1] + track['bbox'][3]) / 2
                    new_cx = (det.bbox[0] + det.bbox[2]) / 2
                    new_cy = (det.bbox[1] + det.bbox[3]) / 2
                    
                    vx = new_cx - prev_cx
                    vy = new_cy - prev_cy
                    
                    # Simple exponential moving average for velocity
                    if track.get('velocity'):
                        old_vx, old_vy = track['velocity']
                        track['velocity'] = (0.7 * old_vx + 0.3 * vx, 0.7 * old_vy + 0.3 * vy)
                    else:
                        track['velocity'] = (vx, vy)
                        
                    track['bbox'] = det.bbox
                    track['last_seen'] = frame_idx
                    track['history'].append((new_cx, new_cy))
                    track['detection'] = det # Update latest detection data (mask/keypoints)
                    
                    # Create Output Entity
                    to = TrackedObject(
                        track_id=track['id'],
                        detection=det,
                        history=track['history'],
                        velocity=track['velocity']
                    )
                    frame_tracks.append(to)
                    
                # Create New Tracks
                for c in unmatched_detections:
                    det = detections[c]
                    new_track = {
                        'id': self.next_id,
                        'bbox': det.bbox,
                        'history': [], # Center points
                        'last_seen': frame_idx,
                        'velocity': None,
                        'detection': det
                    }
                    cx = (det.bbox[0] + det.bbox[2]) / 2
                    cy = (det.bbox[1] + det.bbox[3]) / 2
                    new_track['history'].append((cx, cy))
                    
                    active_tracks.append(new_track)
                    self.next_id += 1
                    
                    to = TrackedObject(
                        track_id=new_track['id'],
                        detection=det,
                        history=new_track['history'],
                        velocity=None
                    )
                    frame_tracks.append(to)
                    
                # Handle Lost Tracks
                # We simply keep them in 'active_tracks' but don't add them to 'frame_tracks' output
                # If they stay unmatched for too long, remove them.
                active_tracks = [t for t in active_tracks if (frame_idx - t['last_seen']) < self.track_buffer]
                
            elif detections:
                # No active tracks, all detections are new
                for det in detections:
                    new_track = {
                        'id': self.next_id,
                        'bbox': det.bbox,
                        'history': [],
                        'last_seen': frame_idx,
                        'velocity': None,
                        'detection': det
                    }
                    cx = (det.bbox[0] + det.bbox[2]) / 2
                    cy = (det.bbox[1] + det.bbox[3]) / 2
                    new_track['history'].append((cx, cy))
                    active_tracks.append(new_track)
                    self.next_id += 1
                    
                    to = TrackedObject(
                        track_id=new_track['id'],
                        detection=det,
                        history=new_track['history'],
                        velocity=None
                    )
                    frame_tracks.append(to)
            
            results[frame_idx] = frame_tracks
            
        print(f"Tracking complete. Total unique IDs: {self.next_id - 1}")
        return results

    def save_result(self, data: Dict[int, List[TrackedObject]], output_path: Path) -> None:
        serializable = {
            str(k): [t.model_dump() for t in v]
            for k, v in data.items()
        }
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def load_result(self, input_path: Path) -> Dict[int, List[TrackedObject]]:
        with open(input_path, 'r') as f:
            raw = json.load(f)
        return {
            int(k): [TrackedObject(**t) for t in v]
            for k, v in raw.items()
        }
