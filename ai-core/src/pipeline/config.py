from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field

class SessionConfig(BaseModel):
    """Configuration for the data session."""
    video_id: str = Field(..., description="Name of the video file (without extension) in data/raw")
    output_dir: str = Field(..., description="Directory name to store run outputs in data/outputs")

class StepConfig(BaseModel):
    """Configuration for a single pipeline step."""
    name: str = Field(..., description="Human-readable name of the step")
    module: str = Field(..., description="Registered module identifier (e.g. 'yolo_v8')")
    enabled: bool = True
    input_source: Literal["memory", "disk"] = Field("memory", description="Where to get input data from")
    input_from_step: Optional[Union[str, List[str]]] = Field(None, description="Name(s) of previous step(s) to fetch input from")
    save_output: bool = Field(False, description="Whether to save the result to disk")
    params: Dict[str, Any] = Field(default_factory=dict, description="Module-specific parameters")

class PipelineConfig(BaseModel):
    """Root configuration for a pipeline execution."""
    session: SessionConfig
    steps: List[StepConfig]
