from typing import Protocol, Iterator, List, Any, Dict, TypeVar
from pathlib import Path
from .entities import VideoSession, FrameData, Detection, TrackedObject

# Generic Input/Output types
T_Input = TypeVar("T_Input", contravariant=True)
T_Output = TypeVar("T_Output", covariant=True)

class IPipelineStep(Protocol[T_Input, T_Output]):
    """
    Common interface for any processing step in the pipeline.
    Enforces hybrid I/O capabilities (Memory & Disk).
    """

    def run(self, input_data: T_Input, config: Dict[str, Any]) -> T_Output:
        """Executes the core logic of the step."""
        ...

    def save_result(self, data: T_Output, output_path: Path) -> None:
        """Serializes the result to disk (JSON/CSV/Pickle)."""
        ...

    def load_result(self, input_path: Path) -> T_Output:
        """Deserializes the result from disk."""
        ...

class IVideoLoader(IPipelineStep[Path, VideoSession]):
    """Interface for loading video files and metadata."""
    
    def load_session(self, video_path: str) -> VideoSession:
        """Loads video metadata and prepares the session."""
        ...
        
    def stream_frames(self, session: VideoSession) -> Iterator[FrameData]:
        """Yields frames from the video session."""
        ...

class IObjectDetector(IPipelineStep[Any, List[Detection]]):
    """Interface for object detection models (e.g. YOLO)."""
    
    def load_model(self, model_path: str) -> None:
        """Loads the model weights."""
        ...
        
    def detect(self, frame: FrameData) -> List[Detection]:
        """Performs detection on a single frame."""
        ...

class ITracker(IPipelineStep[List[Detection], List[TrackedObject]]):
    """Interface for object tracking algorithms."""
    
    def update(self, detections: List[Detection], frame_info: FrameData) -> List[TrackedObject]:
        """Updates tracks with new detections."""
        ...
