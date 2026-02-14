"""
Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ProcessingStatus(str, Enum):
    """Status of video processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    """Response after uploading a video."""
    video_id: str = Field(..., description="Unique identifier for the video")
    status: ProcessingStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Human-readable message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "abc123",
                "status": "pending",
                "message": "Video uploaded successfully. Processing will start shortly."
            }
        }


class StatusResponse(BaseModel):
    """Response for processing status check."""
    video_id: str
    status: ProcessingStatus
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress from 0 to 1")
    current_step: Optional[str] = Field(None, description="Current processing step")
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "abc123",
                "status": "processing",
                "progress": 0.45,
                "current_step": "yolo_detection",
                "message": "Processing YOLO detection...",
                "created_at": "2026-01-29T10:00:00",
                "updated_at": "2026-01-29T10:01:30"
            }
        }


class VideoMetadata(BaseModel):
    """Video metadata."""
    fps: float
    width: int
    height: int
    duration_s: float
    total_frames: int


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float


class FrameDetection(BaseModel):
    """Detection data for a single frame.
    
    For disc tracks: mask is always present (polygon contour), bbox is optional.
    For person tracks (optional): bbox is always present, mask is optional.
    """
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box (optional for disc)")
    mask: Optional[List[List[float]]] = Field(None, description="Polygon points [[x,y], ...]")
    confidence: float = Field(..., ge=0.0, le=1.0)


class Track(BaseModel):
    """Complete track data for an object.
    
    Required tracks:
      - Exactly one track with class_name="frisbee" (the disc). Must have mask + trajectory.
    
    Optional tracks:
      - At most one track with class_name="person" (the athlete). Present only if
        person detection is enabled on the server.
    """
    track_id: int
    class_name: str = Field(..., description="'frisbee' for disc (required), 'person' for athlete (optional)")
    frames: Dict[int, FrameDetection] = Field(..., description="Frame index -> detection")
    trajectory: List[List[float]] = Field(..., description="Center points [[x, y], ...]")
    
    class Config:
        json_schema_extra = {
            "example": {
                "track_id": 1,
                "class_name": "frisbee",
                "frames": {
                    "0": {"mask": [[100, 200], [110, 210], [120, 200]], "confidence": 0.95},
                    "1": {"mask": [[102, 198], [112, 208], [122, 198]], "confidence": 0.93}
                },
                "trajectory": [[110, 205], [112, 203]]
            }
        }


class MetricsSeries(BaseModel):
    """Time series of computed metrics."""
    frames: List[int] = Field(..., description="Frame indices")
    time_s: List[float] = Field(..., description="Timestamps in seconds")
    x_m: List[float] = Field(..., description="X position in meters")
    y_m: List[float] = Field(..., description="Y position in meters")
    height_m: List[float] = Field(..., description="Height above minimum in meters")
    vx_m_s: List[float] = Field(..., description="X velocity in m/s")
    vy_m_s: List[float] = Field(..., description="Y velocity in m/s")
    speed_m_s: List[float] = Field(..., description="Total speed in m/s")
    accel_m_s2: List[float] = Field(..., description="Total acceleration in m/s²")
    kinetic_energy_j: List[float] = Field(..., description="Kinetic energy in Joules")
    potential_energy_j: List[float] = Field(..., description="Potential energy in Joules")
    total_energy_j: List[float] = Field(..., description="Total mechanical energy in Joules")
    power_w: List[float] = Field(..., description="Power in Watts")


class MetricsSummary(BaseModel):
    """Summary statistics of the lift."""
    peak_speed_m_s: float
    peak_power_w: float
    max_height_m: float
    min_height_m: float
    lift_duration_s: float
    total_frames: int


class AnalysisResults(BaseModel):
    """Complete analysis results returned to the client.
    
    Guaranteed content (the mobile app can always rely on):
      - metadata: video properties (fps, resolution, frames, duration)
      - tracks: at least one track with class_name="frisbee" containing mask + trajectory
      - metrics: full time series for the disc (position, velocity, energy, power)
      - summary: peak values (speed, power, height)
    
    Optional content (may or may not be present):
      - A "person" track (only if person detection is enabled server-side)
      - bbox in frame detections (always present for person, optional for disc)
    """
    video_id: str
    status: ProcessingStatus
    metadata: VideoMetadata
    tracks: List[Track] = Field(..., description="Tracked objects (always includes disc)")
    metrics: MetricsSeries = Field(..., description="Time series metrics for the disc")
    summary: MetricsSummary = Field(..., description="Summary statistics")
    processed_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "abc123",
                "status": "completed",
                "metadata": {
                    "fps": 30.0,
                    "width": 1080,
                    "height": 1920,
                    "duration_s": 3.67,
                    "total_frames": 110
                },
                "tracks": [
                    {
                        "track_id": 1,
                        "class_name": "frisbee",
                        "frames": {
                            "0": {"mask": [[100, 200], [110, 210]], "confidence": 0.95}
                        },
                        "trajectory": [[110, 205]]
                    }
                ],
                "metrics": {
                    "frames": [0, 1, 2],
                    "time_s": [0.0, 0.033, 0.067],
                    "height_m": [0.5, 0.52, 0.55],
                    "speed_m_s": [0.0, 0.6, 1.2],
                    "power_w": [0, 150, 320]
                },
                "summary": {
                    "peak_speed_m_s": 4.02,
                    "peak_power_w": 3827,
                    "max_height_m": 1.47,
                    "min_height_m": 0.3,
                    "lift_duration_s": 2.1,
                    "total_frames": 110
                },
                "processed_at": "2026-01-29T10:02:00"
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    video_id: Optional[str] = None


class DeleteResponse(BaseModel):
    """Response after deleting a video."""
    video_id: str
    deleted: bool
    message: str
