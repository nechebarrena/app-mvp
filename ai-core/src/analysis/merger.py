import json
from pathlib import Path
from typing import Dict, Any, List

from domain.ports import IPipelineStep
from domain.entities import Detection

class DetectionMerger(IPipelineStep[List[Dict[int, List[Detection]]], Dict[int, List[Detection]]]):
    """
    Merges detection results from multiple sources (e.g. Pose + Segmentation).
    Assumes inputs are Dict[frame_idx, List[Detection]].
    """

    def run(self, input_data: List[Dict[int, List[Detection]]], config: Dict[str, Any]) -> Dict[int, List[Detection]]:
        """
        Merges a list of detection dictionaries into a single dictionary.
        """
        if not isinstance(input_data, list):
            raise ValueError(f"DetectionMerger expects a List of inputs, got {type(input_data)}")
            
        merged_results: Dict[int, List[Detection]] = {}
        
        print(f"Merging results from {len(input_data)} sources...")
        
        for source_idx, source_data in enumerate(input_data):
            for frame_idx, detections in source_data.items():
                if frame_idx not in merged_results:
                    merged_results[frame_idx] = []
                
                # Append detections
                # Future logic: Could perform IoU matching to merge duplicates?
                # For now, we append everything so the renderer draws everything.
                merged_results[frame_idx].extend(detections)
                
        print(f"Merge complete. Result contains {len(merged_results)} frames.")
        return merged_results

    def save_result(self, data: Dict[int, List[Detection]], output_path: Path) -> None:
        """Serializes the detection map to JSON."""
        serializable_data = {
            str(k): [d.model_dump() for d in v] 
            for k, v in data.items()
        }
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    def load_result(self, input_path: Path) -> Dict[int, List[Detection]]:
        """Deserializes from JSON."""
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
        return {
            int(k): [Detection(**d) for d in v]
            for k, v in raw_data.items()
        }
