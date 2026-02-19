import yaml
import re
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Type, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from .config import PipelineConfig, StepConfig
from domain.ports import IPipelineStep


@dataclass
class VideoMetadata:
    """Video metadata extracted from VideoSession, available to all steps."""
    file_path: str = ""
    fps: float = 30.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    duration_seconds: float = 0.0
    aspect_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PipelineRunner:
    """
    Orchestrates the execution of the pipeline based on a YAML configuration.
    
    Features:
    - Variable substitution: Use ${var_name} in YAML, defined in 'variables' section
    - Disc selection: Can run interactive tool or load from file
    - Video metadata propagation: fps, resolution, etc. available to all steps
    - Optional progress callback: reports step-level progress to caller
    """
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.raw_config: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.video_metadata: VideoMetadata = VideoMetadata()
        
        self._load_and_process_config()
        self._setup_environment()
        self._setup_logging()
        
        self.registry: Dict[str, Type[IPipelineStep]] = {}
        self.step_outputs: Dict[str, Any] = {}
        
        # Optional progress callback: fn(step_name, step_index, total_steps)
        self.on_step_start = None
        
    def register_step(self, module_name: str, step_class: Type[IPipelineStep]):
        """Registers a class to be usable as a pipeline step."""
        self.registry[module_name] = step_class
    
    def _load_and_process_config(self):
        """Load YAML and process variable substitutions."""
        with open(self.config_path, 'r') as f:
            self.raw_config = yaml.safe_load(f)
        
        # Extract variables first
        self.variables = self.raw_config.get('variables', {})
        
        # Substitute variables throughout the config
        processed_config = self._substitute_variables(self.raw_config)
        
        # Parse into Pydantic model
        self.config = PipelineConfig(**processed_config)
    
    def _substitute_variables(self, obj: Any) -> Any:
        """
        Recursively substitute ${var_name} patterns with values from variables.
        """
        if isinstance(obj, str):
            # Find all ${var_name} patterns
            pattern = r'\$\{(\w+)\}'
            
            def replace_var(match):
                var_name = match.group(1)
                if var_name in self.variables:
                    return str(self.variables[var_name])
                # Also check for special built-in variables
                if var_name == 'selection_file' and self.raw_config.get('disc_selection'):
                    return self.raw_config['disc_selection'].get('output_file', '')
                return match.group(0)  # Keep original if not found
            
            return re.sub(pattern, replace_var, obj)
        
        elif isinstance(obj, dict):
            return {k: self._substitute_variables(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [self._substitute_variables(item) for item in obj]
        
        return obj
        
    def _setup_environment(self):
        """Setup project paths and directories."""
        current_path = Path(__file__).resolve()
        self.project_root = None
        for parent in current_path.parents:
            if (parent / "data").exists():
                self.project_root = parent
                break
        
        if not self.project_root:
            raise FileNotFoundError("Could not locate project root (data/ folder not found)")
            
        self.output_dir = self.project_root / "data" / "outputs" / self.config.session.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self):
        """Configure logging to file and console.
        
        Uses a unique logger name per run to avoid handler accumulation
        across multiple API calls (basicConfig is a no-op after the first call).
        """
        log_file = self.output_dir / "pipeline.log"
        self.log_file = log_file

        run_id = self.config.session.output_dir
        logger_name = f"PipelineRunner.{run_id}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if somehow re-initialised
        if not self.logger.handlers:
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
            fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

            # Don't propagate to root logger to avoid double-printing
            self.logger.propagate = False
        
        self.logger.info(f"Pipeline initialized. Run ID: {self.config.session.output_dir}")
        self.logger.info(f"Loaded configuration from {self.config_path}")
        
        # Log variables
        if self.variables:
            self.logger.info(f"Variables defined: {list(self.variables.keys())}")
        
        # Dump config to log for reproducibility
        self.logger.info("Configuration content:")
        self.logger.info(json.dumps(self.config.model_dump(), indent=2, default=str))

    def _handle_disc_selection(self):
        """Handle disc selection based on configuration."""
        if not self.config.disc_selection:
            self.logger.info("No disc_selection config - skipping")
            return
        
        ds_config = self.config.disc_selection
        output_file = Path(ds_config.output_file)
        
        # Resolve relative path
        if not output_file.is_absolute():
            output_file = self.project_root / "ai-core" / ds_config.output_file
        
        if ds_config.mode == "file":
            # Just verify file exists
            if not output_file.exists():
                raise FileNotFoundError(
                    f"Disc selection file not found: {output_file}\n"
                    f"Run with disc_selection.mode='interactive' to create it, or provide existing file."
                )
            self.logger.info(f"Using existing disc selection: {output_file}")
            
        elif ds_config.mode == "interactive":
            self.logger.info("Running interactive disc selection tool...")
            
            # Get video path
            video_path = self.project_root / "data" / "raw" / f"{self.config.session.video_id}.mp4"
            
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found for disc selection: {video_path}")
            
            # Import and run disc selector
            try:
                # Add src to path if needed
                src_path = self.project_root / "ai-core" / "src"
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                
                from tools.disc_selector import DiscSelector, save_selection
                
                print("\n" + "="*60)
                print("DISC SELECTION TOOL")
                print("="*60)
                print(f"Video: {video_path}")
                print(f"Output: {output_file}")
                print("\nInstructions:")
                print("  1. Click 'Centro' button, then click on the disc center")
                print("  2. Click 'Borde' button, then click on the disc edge")
                print("  3. Click 'Aceptar' to save or 'Resetear' to start over")
                print("  4. Press ESC to cancel")
                print("="*60 + "\n")
                
                result = DiscSelector.select_from_video(str(video_path))
                
                if result["accepted"]:
                    # Ensure output directory exists
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    save_selection(result, str(output_file))
                    self.logger.info(f"Disc selection saved: center=({result['center'][0]}, {result['center'][1]}), radius={result['radius']:.1f}")
                else:
                    raise RuntimeError("Disc selection was cancelled by user")
                    
            except ImportError as e:
                raise RuntimeError(f"Could not import disc selector: {e}")
        
        # Store selection file path for use by steps
        self.variables['selection_file'] = str(output_file)

    def _extract_video_metadata(self, video_session):
        """Extract metadata from VideoSession after ingestion."""
        from domain.entities import VideoSession
        
        if isinstance(video_session, VideoSession):
            self.video_metadata = VideoMetadata(
                file_path=video_session.file_path,
                fps=video_session.specs.fps if video_session.specs else 30.0,
                width=video_session.specs.width if video_session.specs else 0,
                height=video_session.specs.height if video_session.specs else 0,
                total_frames=video_session.total_frames,
                duration_seconds=video_session.duration_seconds,
                aspect_ratio=(video_session.specs.width / video_session.specs.height 
                             if video_session.specs and video_session.specs.height > 0 else 1.0)
            )
            self.logger.info(f"Video metadata extracted: {self.video_metadata.width}x{self.video_metadata.height} @ {self.video_metadata.fps:.2f}fps, {self.video_metadata.total_frames} frames")

    def run(self):
        """Executes the pipeline steps."""
        self.logger.info("Starting pipeline execution...")
        
        # Handle disc selection first if configured
        self._handle_disc_selection()
        
        last_result = None
        
        # Count enabled steps for progress
        enabled_steps = [s for s in self.config.steps if s.enabled]
        total_steps = len(enabled_steps)
        current_step_idx = 0
        
        for step_config in self.config.steps:
            if not step_config.enabled:
                self.logger.info(f"Skipping step '{step_config.name}' (disabled)")
                continue
            
            # Report progress via callback
            if self.on_step_start:
                try:
                    self.on_step_start(step_config.name, current_step_idx, total_steps)
                except Exception:
                    pass  # Don't let callback errors break the pipeline
            current_step_idx += 1
                
            self.logger.info(f"Executing step: {step_config.name} (Module: {step_config.module})")
            
            try:
                # 1. Instantiate Step
                step_class = self.registry.get(step_config.module)
                if not step_class:
                    raise ValueError(f"Module '{step_config.module}' not registered.")
                
                step_instance = step_class() 
                
                # 2. Prepare Input
                input_data = None
                
                if step_config.input_source == "disk":
                    if step_config.name == "ingestion":
                        raw_video_path = self.project_root / "data" / "raw" / f"{self.config.session.video_id}.mp4"
                        input_data = raw_video_path
                    else:
                        self.logger.warning("Disk input source requested but path logic not fully implemented.")
                        input_data = self.output_dir 
                
                elif step_config.input_source == "memory":
                    if step_config.input_from_step:
                        source = step_config.input_from_step
                        if isinstance(source, list):
                            input_data = []
                            for src_name in source:
                                if src_name not in self.step_outputs:
                                    raise ValueError(f"Step '{step_config.name}' requires output from '{src_name}' which hasn't run.")
                                input_data.append(self.step_outputs[src_name])
                        else:
                            if source not in self.step_outputs:
                                raise ValueError(f"Step '{step_config.name}' requires output from '{source}' which hasn't run.")
                            input_data = self.step_outputs[source]
                    else:
                        input_data = last_result

                # 3. Build step params with injected context
                self.logger.info(f"Running {step_config.name} with input type: {type(input_data)}")
                
                step_params = dict(step_config.params or {})
                
                # Inject run context
                step_params["_run_id"] = self.config.session.output_dir
                step_params["_run_output_dir"] = str(self.output_dir)
                step_params["_project_root"] = str(self.project_root)
                
                # Inject video metadata (available after ingestion)
                step_params["_video_metadata"] = self.video_metadata.to_dict()
                step_params["_video_fps"] = self.video_metadata.fps
                step_params["_video_width"] = self.video_metadata.width
                step_params["_video_height"] = self.video_metadata.height
                step_params["_video_total_frames"] = self.video_metadata.total_frames
                step_params["_video_duration"] = self.video_metadata.duration_seconds
                step_params["_video_aspect_ratio"] = self.video_metadata.aspect_ratio
                
                # Inject selection file if available
                if self.config.disc_selection:
                    step_params["_selection_file"] = self.variables.get('selection_file', '')
                
                # Run the step
                result = step_instance.run(input_data, step_params)
                
                # 4. Extract video metadata after ingestion
                if step_config.name == "ingestion":
                    self._extract_video_metadata(result)
                
                # 5. Save Output (if requested)
                if step_config.save_output:
                    file_name = f"{step_config.name}_output.json"
                    save_path = self.output_dir / file_name
                    self.logger.info(f"Saving output to {save_path}")
                    step_instance.save_result(result, save_path)
                    
                # 6. Update Memory Context
                self.step_outputs[step_config.name] = result
                last_result = result
                
            except Exception as e:
                self.logger.error(f"Error in step '{step_config.name}': {str(e)}", exc_info=True)
                raise e
                
        self.logger.info("Pipeline execution completed successfully.")
