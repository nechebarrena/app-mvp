import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from domain.ports import IPipelineStep
from domain.entities import TrackedObject, Detection
from analysis.tracking_utils import KalmanFilter2D, GeometryUtils

class TrackerState(Enum):
    TRACKING = "tracking"
    LOST = "lost"

class LiftingSessionOptimizer(IPipelineStep[Dict[int, List[TrackedObject]], Dict[int, List[TrackedObject]]]):
    """
    Robust 3-Layer Tracker:
    A. Athlete Selection (Composite Score).
    B. Disc Tracking (FSM + Kalman + Gating).
    C. Trajectory Cleaning.
    """

    def run(self, input_data: Dict[int, List[TrackedObject]], config: Dict[str, Any]) -> Dict[int, List[TrackedObject]]:
        # NOTE: keep prints for now (useful in notebooks), but also write structured telemetry to disk.
        print("Optimizing lifting session with 3-Layer Architecture (Kalman+FSM)...")

        self._logger = logging.getLogger("LiftingSessionOptimizer")
        self._run_output_dir = Path(config.get("_run_output_dir", ".")).resolve()
        self._debug_enabled = bool(config.get("debug", True))
        self._debug_jsonl_path = self._run_output_dir / "disc_tracker_debug.jsonl"
        if self._debug_enabled:
            try:
                # Fresh file each run
                self._debug_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._debug_jsonl_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"event": "run_start"}, ensure_ascii=False) + "\n")
            except Exception:
                self._logger.exception("Failed to initialize debug telemetry file")
        
        # 1. Organize Input
        detections_by_frame: Dict[int, List[TrackedObject]] = {}
        for frame_idx, objects in input_data.items():
            detections_by_frame[frame_idx] = []
            for obj in objects:
                # Add temp frame index
                obj._temp_frame_idx = frame_idx
                detections_by_frame[frame_idx].append(obj)
                
        # --- LAYER A: ATHLETE SELECTION ---
        athlete_class = config.get("athlete_class_name", "atleta")
        athlete_ref = self._select_target_athlete(detections_by_frame, athlete_class)
        
        if not athlete_ref:
            print(f"  WARNING: No athlete found (class: {athlete_class}). Optimization aborted.")
            return {}
            
        print(f"  Target Athlete Identified. Tracking frames: {len(athlete_ref)}")

        # --- LAYER B: DISC TRACKING ---
        disc_class = config.get("disc_class_name", "discos")
        master_disc_track = self._track_target_disc(detections_by_frame, athlete_ref, disc_class, config)
        
        if master_disc_track:
            print(f"  Target Disc Tracked: {len(master_disc_track)} frames.")
        else:
            print("  WARNING: No disc track established.")

        # --- LAYER C: TRAJECTORY CLEANING ---
        # For now, just combining the outputs. Smoothing can be added here.
        
        # --- REBUILD OUTPUT ---
        final_output: Dict[int, List[TrackedObject]] = {}
        
        # Add Athlete
        for frame_idx, obj in athlete_ref.items():
            # Normalize naming for visualization: treat Pose "person" as domain "atleta"
            if obj.detection.class_name == "person":
                obj.detection.class_name = "atleta"
            if frame_idx not in final_output: final_output[frame_idx] = []
            final_output[frame_idx].append(obj)
            
        # Add Disc
        for obj in master_disc_track:
            f_idx = getattr(obj, '_temp_frame_idx', -1)
            if f_idx >= 0:
                if f_idx not in final_output: final_output[f_idx] = []
                final_output[f_idx].append(obj)
                
        # Cleanup
        for frame_objs in detections_by_frame.values():
            for obj in frame_objs:
                if hasattr(obj, '_temp_frame_idx'): del obj._temp_frame_idx
                
        return final_output

    def _debug_write(self, payload: Dict[str, Any]) -> None:
        if not getattr(self, "_debug_enabled", False):
            return
        try:
            with open(self._debug_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Never fail the pipeline because of debug logging.
            self._logger.exception("Failed to write debug telemetry")

    def _select_target_athlete(self, frames: Dict[int, List[TrackedObject]], class_name: str) -> Dict[int, TrackedObject]:
        """
        Selects the primary athlete using a composite score:
        Score = Persistence + Centrality + Area.
        Returns map: frame_idx -> TrackedObject (best athlete)
        """
        # Group by Track ID
        tracks: Dict[int, List[TrackedObject]] = {}
        for f, objs in frames.items():
            for o in objs:
                if o.detection.class_name == class_name:
                    if o.track_id not in tracks: tracks[o.track_id] = []
                    tracks[o.track_id].append(o)
                    
        if not tracks:
            self._debug_write({"event": "athlete_missing", "athlete_class_name": class_name})
            return {}
        
        # Calculate Frame Center (Approx from all data)
        all_x = []
        for t in tracks.values():
            for o in t:
                all_x.append(GeometryUtils.get_center(o.detection.bbox)[0])
        center_x = np.mean(all_x) if all_x else 0
        
        best_tid = None
        max_score = -float('inf')
        
        total_frames = len(frames)
        
        for tid, history in tracks.items():
            # Features
            persistence = len(history) / total_frames
            
            # Centrality
            dists = [abs(GeometryUtils.get_center(o.detection.bbox)[0] - center_x) for o in history]
            avg_dist = np.mean(dists)
            centrality = 1.0 / (avg_dist + 100.0) # Higher is better
            
            # Area
            areas = [GeometryUtils.get_area(o.detection.bbox) for o in history]
            avg_area = np.mean(areas)
            
            # Composite Score (Weights can be tuned)
            # Prioritize persistence heavily
            score = (persistence * 0.5) + (centrality * 10000.0 * 0.3) + (avg_area * 0.0001 * 0.2)
            
            if score > max_score:
                max_score = score
                best_tid = tid
                
        if best_tid is None: return {}

        # Telemetry: athlete selection summary (top few tracks)
        try:
            ranked = []
            for tid, history in tracks.items():
                persistence = len(history) / max(1, total_frames)
                dists = [abs(GeometryUtils.get_center(o.detection.bbox)[0] - center_x) for o in history]
                avg_dist = float(np.mean(dists)) if dists else 0.0
                areas = [GeometryUtils.get_area(o.detection.bbox) for o in history]
                avg_area = float(np.mean(areas)) if areas else 0.0
                centrality = float(1.0 / (avg_dist + 100.0))
                score = float((persistence * 0.5) + (centrality * 10000.0 * 0.3) + (avg_area * 0.0001 * 0.2))
                ranked.append({"track_id": int(tid), "frames": int(len(history)), "persistence": float(persistence), "avg_dist": avg_dist, "avg_area": avg_area, "score": score})
            ranked.sort(key=lambda x: x["score"], reverse=True)
            self._debug_write({
                "event": "athlete_selected",
                "athlete_class_name": class_name,
                "chosen_track_id": int(best_tid),
                "total_frames": int(total_frames),
                "candidates": ranked[:5],
            })
        except Exception:
            self._logger.exception("Failed athlete selection telemetry")
        
        # Convert list to map
        best_track_map = {}
        for o in tracks[best_tid]:
            best_track_map[o._temp_frame_idx] = o

        # Forward-fill missing frames so ROI-based logic can work even if Pose misses intermittently.
        # IMPORTANT: do NOT forward-fill keypoints; otherwise the skeleton will appear "stuck" on screen.
        all_frames = sorted(frames.keys())
        last_obj: Optional[TrackedObject] = None
        filled: Dict[int, TrackedObject] = {}
        for f in all_frames:
            if f in best_track_map:
                last_obj = best_track_map[f]
                filled[f] = last_obj
            else:
                if last_obj is not None:
                    # Create a lightweight copy for ROI usage but remove keypoints to avoid frozen skeletons.
                    det = last_obj.detection.model_copy(update={"keypoints": None})
                    filled[f] = last_obj.model_copy(update={"detection": det})

        return filled

    def _track_target_disc(self, frames: Dict[int, List[TrackedObject]], 
                          athlete_ref: Dict[int, TrackedObject], 
                          class_name: str, config: Dict[str, Any]) -> List[TrackedObject]:
        """
        Tracks the disc using Kalman Filter, FSM, and Athlete ROI Gating.
        """
        
        # Params
        seed_frames = 30
        lost_enter = config.get("lost_enter_frames", 3)
        lost_stop = config.get("lost_stop_frames", 60)
        ratio_min_tracking = config.get("ratio_min_tracking", 0.65)
        ratio_min_lost = config.get("ratio_min_lost", 0.50)
        area_ratio_min_tracking = config.get("area_ratio_min_tracking", 0.55)
        area_ratio_min_lost = config.get("area_ratio_min_lost", 0.40)
        
        # 1. Find Seed
        sorted_indices = sorted(frames.keys())
        seed_limit_idx = min(len(sorted_indices), seed_frames)
        seed_indices = sorted_indices[:seed_limit_idx]
        
        seed_obj, seed_frame_idx = self._find_robust_seed(frames, seed_indices, athlete_ref, class_name)

        self._debug_write({
            "event": "seed_selected" if seed_obj else "seed_missing",
            "disc_class_name": class_name,
            "seed_frames_window": int(seed_frames),
            "seed_frame": int(seed_frame_idx),
            "seed_bbox": list(seed_obj.detection.bbox) if seed_obj else None,
            "seed_ratio": float(GeometryUtils.get_aspect_ratio(seed_obj.detection.bbox)) if seed_obj else None,
            "seed_area": float(GeometryUtils.get_area(seed_obj.detection.bbox)) if seed_obj else None,
            "athlete_ref_frames": int(len(athlete_ref)),
        })
        
        if not seed_obj:
            return []
            
        # Initialize Tracking State
        state = TrackerState.TRACKING
        lost_counter = 0
        
        center_init = GeometryUtils.get_center(seed_obj.detection.bbox)
        kf = KalmanFilter2D(
            x=center_init[0], y=center_init[1],
            process_noise=config.get("kalman_process_noise", 1e-3),
            measurement_noise=config.get("kalman_measurement_noise", 1e-1)
        )
        
        ref_area = GeometryUtils.get_area(seed_obj.detection.bbox)
        
        master_track = []
        
        # Assign Master ID
        master_id = 777
        seed_obj.track_id = master_id
        master_track.append(seed_obj)
        
        # Start iterating from seed frame + 1
        start_pos = sorted_indices.index(seed_frame_idx)
        
        for i in range(start_pos + 1, len(sorted_indices)):
            f_idx = sorted_indices[i]
            
            # Predict
            pred_x, pred_y = kf.predict()
            
            # Get Athlete ROI
            athlete = athlete_ref.get(f_idx)
            roi_box = self._get_dynamic_roi(athlete, expand=(state == TrackerState.LOST))
            
            # Get Candidates
            candidates = [o for o in frames[f_idx] if o.detection.class_name == class_name]
            
            # Gating & Scoring
            best_cand = None
            min_cost = float('inf')
            
            search_radius = self._get_dynamic_radius(athlete, state)
            
            reject_counts = {"roi": 0, "ratio": 0, "radius": 0, "area_ratio": 0}
            best_snapshot = None

            for cand in candidates:
                # 1. ROI Filter
                c_center = GeometryUtils.get_center(cand.detection.bbox)
                if roi_box:
                    if not (roi_box[0] <= c_center[0] <= roi_box[2] and roi_box[1] <= c_center[1] <= roi_box[3]):
                        reject_counts["roi"] += 1
                        continue # Outside ROI

                # 1.5 Shape gate (squareness)
                ratio = GeometryUtils.get_aspect_ratio(cand.detection.bbox)
                if state == TrackerState.TRACKING and ratio < ratio_min_tracking:
                    reject_counts["ratio"] += 1
                    continue
                if state == TrackerState.LOST and ratio < ratio_min_lost:
                    reject_counts["ratio"] += 1
                    continue
                
                # 2. Distance Filter (Gating)
                dist = np.sqrt((c_center[0] - pred_x)**2 + (c_center[1] - pred_y)**2)
                if dist > search_radius:
                    reject_counts["radius"] += 1
                    continue
                    
                # 3. Cost Function
                # Area Similarity
                c_area = GeometryUtils.get_area(cand.detection.bbox)
                area_ratio = min(c_area, ref_area) / max(c_area, ref_area)
                if state == TrackerState.TRACKING and area_ratio < area_ratio_min_tracking:
                    reject_counts["area_ratio"] += 1
                    continue
                if state == TrackerState.LOST and area_ratio < area_ratio_min_lost:
                    reject_counts["area_ratio"] += 1
                    continue
                
                # Cost weights
                cost = (dist * 1.0) + (1000.0 * (1.0 - area_ratio)) + (500.0 * (1.0 - ratio))
                
                if cost < min_cost:
                    min_cost = cost
                    best_cand = cand
                    best_snapshot = {
                        "bbox": list(cand.detection.bbox),
                        "dist": float(dist),
                        "ratio": float(ratio),
                        "area_ratio": float(area_ratio),
                        "cost": float(cost),
                    }
            
            # Decision
            cost_threshold = 300.0 if state == TrackerState.TRACKING else 400.0
            
            accepted = bool(best_cand and min_cost < cost_threshold)

            self._debug_write({
                "event": "frame_eval",
                "frame": int(f_idx),
                "state": state.name,
                "pred": [float(pred_x), float(pred_y)],
                "roi": list(roi_box) if roi_box else None,
                "radius": float(search_radius),
                "candidates_total": int(len(candidates)),
                "reject_counts": reject_counts,
                "best": best_snapshot,
                "accepted": accepted,
                "cost_threshold": float(cost_threshold),
                "ref_area": float(ref_area),
            })

            if accepted:
                # Measurement Accepted
                meas_center = GeometryUtils.get_center(best_cand.detection.bbox)
                kf.update(meas_center)
                
                # Update Ref Area (Slowly)
                meas_area = GeometryUtils.get_area(best_cand.detection.bbox)
                ref_area = 0.9 * ref_area + 0.1 * meas_area
                
                # State Update
                if state == TrackerState.LOST:
                    print(f"  Frame {f_idx}: REACQUIRED disc!")
                state = TrackerState.TRACKING
                lost_counter = 0
                
                best_cand.track_id = master_id
                master_track.append(best_cand)
                
            else:
                # Measurement Lost
                lost_counter += 1
                if state == TrackerState.TRACKING and lost_counter >= lost_enter:
                    state = TrackerState.LOST
                    print(f"  Frame {f_idx}: Tracking LOST (Gap start).")
                    self._debug_write({"event": "state_change", "frame": int(f_idx), "state": state.name, "lost_counter": int(lost_counter)})
                    
                if state == TrackerState.LOST and lost_counter >= lost_stop:
                    print(f"  Frame {f_idx}: Tracking STOPPED (Lost too long).")
                    self._debug_write({"event": "state_change", "frame": int(f_idx), "state": "STOPPED", "lost_counter": int(lost_counter)})
                    break
                    
                # In LOST state, we don't add object to track, but we keep predicting
                # Optionally, add a placeholder object?
                pass
                
        return master_track

    def _find_robust_seed(self, frames: Dict[int, List[TrackedObject]], frame_indices: List[int], 
                         athlete_ref: Dict[int, TrackedObject], class_name: str) -> Tuple[Optional[TrackedObject], int]:
        """
        Finds the best seed object in the first N frames using ROI and Shape.
        """
        best_obj = None
        best_f = -1
        max_score = -float('inf')
        
        for f in frame_indices:
            athlete = athlete_ref.get(f)
            roi = self._get_dynamic_roi(athlete, expand=False)
            athlete_center = None
            if athlete is not None:
                athlete_center = GeometryUtils.get_center(athlete.detection.bbox)
            
            for obj in frames[f]:
                if obj.detection.class_name != class_name: continue
                
                center = GeometryUtils.get_center(obj.detection.bbox)
                
                # 1. ROI Check
                # If we have ROI, enforce it. If we don't (pose missing), allow candidates.
                if roi is not None:
                    if not (roi[0] <= center[0] <= roi[2] and roi[1] <= center[1] <= roi[3]):
                        continue
                    
                # 2. Score
                area = GeometryUtils.get_area(obj.detection.bbox)
                ratio = GeometryUtils.get_aspect_ratio(obj.detection.bbox)
                
                # Prefer VERY square discs over simply large ones.
                # Use ratio^4 to strongly penalize perspective-ellipses on the floor.
                score = area * (ratio ** 4)

                # If we know athlete location, prefer discs closer to athlete (avoid random floor plates)
                if athlete_center is not None:
                    dist = np.sqrt((center[0] - athlete_center[0])**2 + (center[1] - athlete_center[1])**2)
                    score -= 0.5 * dist
                
                if score > max_score:
                    max_score = score
                    best_obj = obj
                    best_f = f
                    
        return best_obj, best_f

    def _get_dynamic_roi(self, athlete: Optional[TrackedObject], expand: bool) -> Optional[Tuple[float, float, float, float]]:
        """Returns (x1, y1, x2, y2) ROI based on athlete position."""
        if not athlete: return None
        
        ax1, ay1, ax2, ay2 = athlete.detection.bbox
        w = ax2 - ax1
        h = ay2 - ay1
        cx = (ax1 + ax2) / 2
        cy = (ay1 + ay2) / 2
        
        # Horizontal: athlete width +/- margin (disc can be far from torso in side view)
        margin_x = (1.2 * w) if not expand else (2.0 * w)
        # Vertical: allow bar to rise above torso; keep bounded but generous
        margin_y = (0.6 * h) if not expand else (1.0 * h)
        
        roi_x1 = cx - margin_x
        roi_x2 = cx + margin_x
        roi_y1 = ay1 - margin_y
        roi_y2 = ay2 + margin_y
        
        return (roi_x1, roi_y1, roi_x2, roi_y2)

    def _get_dynamic_radius(self, athlete: Optional[TrackedObject], state: TrackerState) -> float:
        """Returns search radius based on athlete size."""
        if not athlete: return 100.0 # Fallback
        
        w = athlete.detection.bbox[2] - athlete.detection.bbox[0]
        
        base_radius = 0.4 * w # 40% of athlete width
        
        if state == TrackerState.LOST:
            return base_radius * 2.0
        return base_radius
