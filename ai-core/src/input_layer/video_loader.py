import json
import cv2
from pathlib import Path
from typing import Dict, Any, Optional

from domain.ports import IPipelineStep
from domain.entities import VideoSession, CameraSpecs

class VideoLoader(IPipelineStep[Any, VideoSession]):
    """
    Pipeline step that initializes a VideoSession from a raw video file.
    Does NOT load frames into memory, just metadata.
    """

    def run(self, input_data: Any, config: Dict[str, Any]) -> VideoSession:
        """
        Loads the video session.
        If input_data is None (start of pipeline), looks for video_id in config context.
        Note: The 'config' dict passed here is the 'params' from the YAML step config.
        We need access to the GLOBAL session config (video_id) to know what to load.
        
        Wait, standard IPipelineStep receives 'params'. It doesn't receive the full global config object.
        Design Fix: The Runner should probably inject the video_id or path into input_data if it's the first step?
        OR: We rely on the Runner passing the video path as input_data for the first step (as discussed in runner.py implementation).
        
        Let's assume input_data is the Path to the video file if passed by Runner.
        """
        
        video_path = None
        if isinstance(input_data, (str, Path)):
            video_path = Path(input_data)
        
        if not video_path or not video_path.exists():
            raise ValueError(f"VideoLoader received invalid input path: {input_data}")

        print(f"Loading video session from: {video_path}")
        
        # 1. Look for Sidecar JSON
        json_path = video_path.with_suffix('.json')
        specs = None
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                meta = json.load(f)
                # Parse technical specs
                tech = meta.get("technical", {})
                cam = meta.get("camera_specs", {})
                
                specs = CameraSpecs(
                    width=tech.get("width", 0),
                    height=tech.get("height", 0),
                    fps=tech.get("fps", 0.0),
                    focal_length=cam.get("focal_length_mm"),
                    device_model=cam.get("device_model")
                )
                total_frames = tech.get("frame_count", 0)
                duration = tech.get("duration_seconds", 0.0)
        else:
            # Fallback: Read with OpenCV directly if no JSON (Should not happen if scanner run)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            specs = CameraSpecs(
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS)
            )
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / specs.fps if specs.fps > 0 else 0
            cap.release()

        # 2. Create Session Object
        session = VideoSession(
            file_path=str(video_path.absolute()),
            specs=specs,
            total_frames=total_frames,
            duration_seconds=duration
        )
        
        return session

    def save_result(self, data: VideoSession, output_path: Path) -> None:
        """Serializes the VideoSession object to JSON."""
        with open(output_path, 'w') as f:
            f.write(data.model_dump_json(indent=2))

    def load_result(self, input_path: Path) -> VideoSession:
        """Deserializes a VideoSession from JSON."""
        with open(input_path, 'r') as f:
            raw = json.load(f)
        return VideoSession(**raw)
