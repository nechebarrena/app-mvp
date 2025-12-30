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
    bbox: Tuple[float, float, float, float] # (x_min, y_min, x_max, y_max) normalized or pixels? 
    # Let's standardize on PIXELS for this project to avoid confusion, or Normalized. 
    # YOLO returns pixels usually. Let's document: PIXELS [x1, y1, x2, y2]
    mask: Optional[Any] = None # Placeholder for segmentation mask (numpy array or RLE)

    class Config:
        arbitrary_types_allowed = True

class TrackedObject(BaseModel):
    """An object being tracked across frames."""
    track_id: int
    detection: Detection
    history: List[Tuple[float, float]] = Field(default_factory=list) # Center points (x, y)
    velocity: Optional[Tuple[float, float]] = None

# --- Runtime Entities (Dataclasses for internal flow) ---

@dataclass
class FrameData:
    """Represents a single frame in the pipeline."""
    frame_index: int
    timestamp: float
    image: np.ndarray # The actual image data (OpenCV BGR)
    original_shape: Tuple[int, int]
    
    def __repr__(self):
        return f"FrameData(idx={self.frame_index}, ts={self.timestamp:.2f}, shape={self.image.shape})"

@dataclass
class VideoSession:
    """Represents a loaded video session ready for processing."""
    file_path: str
    specs: CameraSpecs
    total_frames: int
    duration_seconds: float
    
    # We might add a generator or iterator here later, but for the entity 
    # it just holds metadata about the session.


