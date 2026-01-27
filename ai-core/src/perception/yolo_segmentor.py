import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from ultralytics import YOLO

from domain.ports import IPipelineStep
from domain.entities import VideoSession, Detection

class YoloSegmentor(IPipelineStep[VideoSession, Dict[int, List[Detection]]]):
    """
    Pipeline step that runs YOLO segmentation on a VideoSession.
    """

    def run(self, input_data: VideoSession, config: Dict[str, Any]) -> Dict[int, List[Detection]]:
        """
        Runs inference on the video frames.
        Returns a map of {frame_index: [List of Detections]}.
        """
        video_path = Path(input_data.file_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        model_name = config.get("model_path", "yolov8n-seg.pt") # Default to nano if not specified
        conf_threshold = config.get("conf_threshold", 0.5)
        
        # Resolve model path relative to project root if it's a local file
        # We assume if it starts with 'data/', it's relative to project root.
        # But we need to find project root again or assume CWD is project root?
        # The runner sets CWD to project root usually? No, runner is run from anywhere.
        # Let's check if the path exists as is, if not, try to prepend project root logic?
        # For now, let's assume the user configures the FULL relative path from where the script is run.
        
        print(f"Loading YOLO model: {model_name}")
        model = YOLO(model_name)
        
        results_map: Dict[int, List[Detection]] = {}
        
        # Open Video
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        print(f"Starting inference on {video_path.name}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run Inference
            # stream=True is efficient for videos
            # verbose=False to keep logs clean
            results = model.predict(frame, conf=conf_threshold, verbose=False, task='segment')
            
            frame_detections: List[Detection] = []
            
            for r in results:
                # r.boxes contains bounding boxes
                # r.masks contains segmentation masks
                
                boxes = r.boxes
                masks = r.masks
                
                if boxes is None:
                    continue
                    
                for i, box in enumerate(boxes):
                    # Coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist() # Pixels
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    # Mask processing
                    mask_data = None
                    if masks is not None and len(masks) > i:
                        # Get mask for this object
                        # masks.xy is a list of points (polygon) normalized or pixels?
                        # Ultralytics docs: masks.xy is list of segments in pixels
                        # We can store the polygon points for lightweight storage
                        # instead of the full bitmap.
                        mask_data = masks.xy[i].tolist() # List of [x, y] points
                        
                    detection = Detection(
                        class_id=cls,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        mask=mask_data,
                        source="yolo"
                    )
                    frame_detections.append(detection)
            
            if frame_detections:
                results_map[frame_idx] = frame_detections
                
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx}...", end='\r')
                
        cap.release()
        print(f"\nInference complete. Processed {frame_idx} frames.")
        
        return results_map

    def save_result(self, data: Dict[int, List[Detection]], output_path: Path) -> None:
        """Serializes the detection map to JSON."""
        # Convert dictionary keys (int) to str for JSON compatibility,
        # and Pydantic models to dicts.
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
            
        # Convert back keys to int and dicts to Detection objects
        return {
            int(k): [Detection(**d) for d in v]
            for k, v in raw_data.items()
        }

