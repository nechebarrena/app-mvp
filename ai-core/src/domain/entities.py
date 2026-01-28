from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import numpy as np
from pydantic import BaseModel, Field

# --- Base Data Structures (Pydantic for serialization/contracts) ---

class CameraSpecs(BaseModel):
    """Specifications of the camera used to capture the video."""
    width: int
    height: int
    fps: float
    focal_length: Optional[float] = None
    device_model: Optional[str] = None # e.g. "iPhone 14"

class Detection(BaseModel):
    """A single detected object in a frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float] # (x_min, y_min, x_max, y_max)
    mask: Optional[Any] = None # Placeholder for segmentation mask (numpy array or RLE)
    keypoints: Optional[List[List[float]]] = None # List of [x, y, conf] for pose estimation
    # Optional provenance / geometry (used by hybrid pipelines)
    source: Optional[str] = None # e.g. "yolo", "geom", "pose", "fused"
    radius_px: Optional[float] = None # circle-like radius estimate in pixels (if applicable)
    shape_score: Optional[float] = None # e.g. circularity [0..1] for geom detections
    debug: Optional[Any] = None # free-form debug payload (JSON-serializable dict recommended)

    class Config:
        arbitrary_types_allowed = True

class TrackedObject(BaseModel):
    """An object being tracked across frames."""
    track_id: int
    detection: Detection
    history: List[Tuple[float, float]] = Field(default_factory=list) # Center points (x, y)
    velocity: Optional[Tuple[float, float]] = None
    smoothed_position: Optional[Tuple[float, float]] = None  # Position after trajectory smoothing

# --- Runtime Entities ---

@dataclass
class FrameData:
    """Represents a single frame in the pipeline."""
    frame_index: int
    timestamp: float
    image: np.ndarray # The actual image data (OpenCV BGR)
    original_shape: Tuple[int, int]
    
    def __repr__(self):
        return f"FrameData(idx={self.frame_index}, ts={self.timestamp:.2f}, shape={self.image.shape})"

class VideoSession(BaseModel):
    """Represents a loaded video session ready for processing."""
    file_path: str
    specs: CameraSpecs
    total_frames: int
    duration_seconds: float
