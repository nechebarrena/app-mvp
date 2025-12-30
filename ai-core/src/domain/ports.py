from typing import Protocol, Iterator, List, Optional
from pathlib import Path
import numpy as np
from .entities import VideoSession, FrameData, Detection, TrackedObject

class IVideoLoader(Protocol):
    """Interface for loading video files and metadata."""
    
    def load_session(self, video_path: str) -> VideoSession:
        """Loads video metadata and prepares the session."""
        ...
        
    def stream_frames(self, session: VideoSession) -> Iterator[FrameData]:
        """Yields frames from the video session."""
        ...

class IObjectDetector(Protocol):
    """Interface for object detection models (e.g. YOLO)."""
    
    def load_model(self, model_path: str) -> None:
        """Loads the model weights."""
        ...
        
    def detect(self, frame: FrameData) -> List[Detection]:
        """Performs detection on a single frame."""
        ...

class ITracker(Protocol):
    """Interface for object tracking algorithms."""
    
    def update(self, detections: List[Detection], frame_info: FrameData) -> List[TrackedObject]:
        """Updates tracks with new detections."""
        ...


