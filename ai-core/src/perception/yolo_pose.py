import json
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional
from ultralytics import YOLO

from domain.ports import IPipelineStep
from domain.entities import VideoSession, Detection

class YoloPoseDetector(IPipelineStep[VideoSession, Dict[int, List[Detection]]]):
    """
    Pipeline step that runs YOLO Pose estimation on a VideoSession.
    """

    def run(self, input_data: VideoSession, config: Dict[str, Any]) -> Dict[int, List[Detection]]:
        """
        Runs inference on the video frames.
        Returns a map of {frame_index: [List of Detections]}.
        """
        video_path = Path(input_data.file_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        model_name = config.get("model_path", "yolov8n-pose.pt") # Default to nano pose
        conf_threshold = config.get("conf_threshold", 0.5)
        
        print(f"Loading YOLO Pose model: {model_name}")
        model = YOLO(model_name)
        
        results_map: Dict[int, List[Detection]] = {}
        
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        print(f"Starting pose estimation on {video_path.name}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run Inference
            results = model.predict(frame, conf=conf_threshold, verbose=False, task='pose')
            
            frame_detections: List[Detection] = []
            
            for r in results:
                boxes = r.boxes
                keypoints = r.keypoints
                
                if boxes is None:
                    continue
                    
                for i, box in enumerate(boxes):
                    # Coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    # Keypoints extraction
                    # keypoints.data is [N, 17, 3] (x, y, conf)
                    kpts_data = None
                    if keypoints is not None and len(keypoints.data) > i:
                        kpts_data = keypoints.data[i].tolist()
                    
                    detection = Detection(
                        class_id=cls,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        keypoints=kpts_data,
                        source="pose"
                    )
                    frame_detections.append(detection)
            
            if frame_detections:
                results_map[frame_idx] = frame_detections
                
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx}...", end='\r')
                
        cap.release()
        print(f"\nPose estimation complete. Processed {frame_idx} frames.")
        
        return results_map

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

