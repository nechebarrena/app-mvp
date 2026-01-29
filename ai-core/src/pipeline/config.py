from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field


class SessionConfig(BaseModel):
    """Configuration for the data session."""
    video_id: str = Field(..., description="Name of the video file (without extension) in data/raw")
    output_dir: str = Field(..., description="Directory name to store run outputs in data/outputs")


class DiscSelectionConfig(BaseModel):
    """Configuration for disc selection (initial position/size)."""
    mode: Literal["interactive", "file"] = Field(
        "file", 
        description="'interactive' runs GUI tool, 'file' loads from existing JSON"
    )
    output_file: str = Field(
        "../data/outputs/disc_selection.json",
        description="Path to save/load selection JSON"
    )


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
    # Variables for DRY configuration - can be referenced as ${var_name}
    variables: Dict[str, Any] = Field(default_factory=dict, description="Shared variables for substitution")
    
    # Disc selection configuration
    disc_selection: Optional[DiscSelectionConfig] = Field(
        None, 
        description="Configuration for disc selection tool"
    )
    
    # Core configuration
    session: SessionConfig
    steps: List[StepConfig]
