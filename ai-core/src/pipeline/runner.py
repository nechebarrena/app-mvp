import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any, Type, Optional
from datetime import datetime

from .config import PipelineConfig, StepConfig
from domain.ports import IPipelineStep

class PipelineRunner:
    """
    Orchestrates the execution of the pipeline based on a YAML configuration.
    """
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._load_config()
        self._setup_environment()
        self._setup_logging()
        self.registry: Dict[str, Type[IPipelineStep]] = {}
        # Stores the output of each step by step name
        self.step_outputs: Dict[str, Any] = {}
        
    def register_step(self, module_name: str, step_class: Type[IPipelineStep]):
        """Registers a class to be usable as a pipeline step."""
        self.registry[module_name] = step_class
        
    def _load_config(self):
        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        self.config = PipelineConfig(**raw_config)
        
    def _setup_environment(self):
        # Resolve project root (assuming runner is in src/pipeline)
        # We need to find the 'data' folder.
        # This logic mimics the scanner's root finding for robustness.
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
        log_file = self.output_dir / "pipeline.log"
        
        # Configure logging to file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("PipelineRunner")
        
        self.logger.info(f"Pipeline initialized. Run ID: {self.config.session.output_dir}")
        self.logger.info(f"Loaded configuration from {self.config_path}")
        
        # Dump config to log for reproducibility
        self.logger.info("Configuration content:")
        self.logger.info(json.dumps(self.config.model_dump(), indent=2, default=str))

    def run(self):
        """Executes the pipeline steps."""
        self.logger.info("Starting pipeline execution...")
        
        last_result = None # Keep track of last result for default behavior
        
        for step_config in self.config.steps:
            if not step_config.enabled:
                self.logger.info(f"Skipping step '{step_config.name}' (disabled)")
                continue
                
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
                    # Logic for disk loading remains mostly manual for now or specific to Ingestion
                    if step_config.name == "ingestion":
                        raw_video_path = self.project_root / "data" / "raw" / f"{self.config.session.video_id}.mp4"
                        input_data = raw_video_path
                    else:
                         self.logger.warning("Disk input source requested but path logic not fully implemented. Passing output dir.")
                         input_data = self.output_dir 
                
                elif step_config.input_source == "memory":
                    # --- NEW LOGIC: Input Selection ---
                    if step_config.input_from_step:
                        # If specific step(s) requested
                        source = step_config.input_from_step
                        if isinstance(source, list):
                            # List of sources -> List of inputs
                            input_data = []
                            for src_name in source:
                                if src_name not in self.step_outputs:
                                    raise ValueError(f"Step '{step_config.name}' requires output from '{src_name}' which hasn't run or produced output.")
                                input_data.append(self.step_outputs[src_name])
                        else:
                            # Single source
                            if source not in self.step_outputs:
                                raise ValueError(f"Step '{step_config.name}' requires output from '{source}' which hasn't run.")
                            input_data = self.step_outputs[source]
                    else:
                        # Default: Use output from immediately previous step
                        input_data = last_result

                # 3. Run Step
                self.logger.info(f"Running {step_config.name} with input type: {type(input_data)}")
                # Inject run context into every step for consistent debug artifact placement.
                # These keys are reserved (prefixed) to avoid collision with user params.
                step_params = dict(step_config.params or {})
                step_params["_run_id"] = self.config.session.output_dir
                step_params["_run_output_dir"] = str(self.output_dir)
                step_params["_project_root"] = str(self.project_root)
                result = step_instance.run(input_data, step_params)
                
                # 4. Save Output (if requested)
                if step_config.save_output:
                    file_name = f"{step_config.name}_output.json" # Default naming
                    save_path = self.output_dir / file_name
                    self.logger.info(f"Saving output to {save_path}")
                    step_instance.save_result(result, save_path)
                    
                # 5. Update Memory Context
                self.step_outputs[step_config.name] = result
                last_result = result
                
            except Exception as e:
                self.logger.error(f"Error in step '{step_config.name}': {str(e)}", exc_info=True)
                raise e
                
        self.logger.info("Pipeline execution completed successfully.")
