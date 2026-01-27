import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from domain.ports import IPipelineStep
from domain.entities import TrackedObject

class TrajectoryCleaner(IPipelineStep[Dict[int, List[TrackedObject]], Dict[int, List[TrackedObject]]]):
    """
    Refines tracking results: filters noise and interpolates gaps.
    """

    def run(self, input_data: Dict[int, List[TrackedObject]], config: Dict[str, Any]) -> Dict[int, List[TrackedObject]]:
        
        min_duration = config.get("min_duration_frames", 10)
        max_gap = config.get("max_gap_frames", 5)
        
        print(f"Cleaning trajectories (Min duration: {min_duration}, Max Gap: {max_gap})...")
        
        # 1. Reorganize by Track ID
        # {track_id: {frame_idx: TrackedObject}}
        tracks_by_id: Dict[int, Dict[int, TrackedObject]] = {}
        
        for frame_idx, track_list in input_data.items():
            for obj in track_list:
                if obj.track_id not in tracks_by_id:
                    tracks_by_id[obj.track_id] = {}
                tracks_by_id[obj.track_id][frame_idx] = obj
                
        cleaned_results: Dict[int, List[TrackedObject]] = {}
        
        # 2. Process each track
        valid_tracks = 0
        interpolated_frames = 0
        
        for tid, frames_map in tracks_by_id.items():
            sorted_frames = sorted(frames_map.keys())
            if not sorted_frames:
                continue
                
            start_f = sorted_frames[0]
            end_f = sorted_frames[-1]
            duration = end_f - start_f + 1
            
            # Filter by duration
            if duration < min_duration:
                continue
            
            valid_tracks += 1
            
            # Fill Gaps (Linear Interpolation)
            current_f = start_f
            while current_f <= end_f:
                if current_f in frames_map:
                    # Frame exists, add to result
                    if current_f not in cleaned_results:
                        cleaned_results[current_f] = []
                    cleaned_results[current_f].append(frames_map[current_f])
                    current_f += 1
                else:
                    # Gap detected!
                    # Find next existing frame
                    next_f = current_f + 1
                    while next_f <= end_f and next_f not in frames_map:
                        next_f += 1
                        
                    gap_size = next_f - current_f
                    
                    if gap_size <= max_gap:
                        # Interpolate
                        prev_obj = frames_map[current_f - 1]
                        next_obj = frames_map[next_f]
                        
                        # Interpolate all frames in gap
                        for i in range(gap_size):
                            interp_idx = current_f + i
                            ratio = (i + 1) / (gap_size + 1)
                            
                            # Interpolate BBox
                            # simple linear: prev + ratio * (next - prev)
                            new_bbox = tuple(
                                p + ratio * (n - p) 
                                for p, n in zip(prev_obj.detection.bbox, next_obj.detection.bbox)
                            )
                            
                            # Create synthetic object (clone prev detection but update bbox)
                            # Note: Masks/Keypoints are hard to interpolate linearly.
                            # We keep them None or copy from nearest neighbor?
                            # Let's copy from prev for simplicity but update bbox.
                            synth_det = prev_obj.detection.model_copy(update={"bbox": new_bbox})
                            
                            synth_obj = TrackedObject(
                                track_id=tid,
                                detection=synth_det,
                                history=prev_obj.history, # Could update history too
                                velocity=prev_obj.velocity
                            )
                            
                            if interp_idx not in cleaned_results:
                                cleaned_results[interp_idx] = []
                            cleaned_results[interp_idx].append(synth_obj)
                            interpolated_frames += 1
                            
                        current_f = next_f
                    else:
                        # Gap too large, don't interpolate
                        current_f = next_f

        print(f"Cleaning complete. Kept {valid_tracks} tracks. Interpolated {interpolated_frames} frames.")
        return cleaned_results

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
