"""
Background task processing for video analysis.

Integrates with the existing pipeline to run analysis asynchronously.
"""

import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import yaml
import pandas as pd
import cv2

from .models import ProcessingStatus
from .storage import get_storage, StorageManager


# Pipeline progress mapping (step name -> progress fraction)
STEP_PROGRESS = {
    "ingestion": 0.05,
    "yolo_coco_detection": 0.30,
    "yolo_pose": 0.45,
    "detection_filter": 0.55,
    "disc_tracking": 0.70,
    "track_refiner": 0.80,
    "metrics_calculator": 0.90,
    "complete": 1.0
}


class PipelineProgressCallback:
    """Callback to update job progress as pipeline runs."""
    
    def __init__(self, storage: StorageManager, video_id: str):
        self.storage = storage
        self.video_id = video_id
    
    def on_step_start(self, step_name: str):
        """Called when a pipeline step starts."""
        progress = STEP_PROGRESS.get(step_name, 0.5)
        self.storage.update_job(
            self.video_id,
            status=ProcessingStatus.PROCESSING,
            progress=progress,
            current_step=step_name,
            message=f"Running {step_name}..."
        )
    
    def on_step_complete(self, step_name: str):
        """Called when a pipeline step completes."""
        pass


def create_api_pipeline_config(
    video_path: str,
    output_dir: str,
    selection_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a pipeline configuration for API processing.
    
    This is a simplified version of full_analysis.yaml optimized for API use.
    """
    # Extract video_id from path (for API, we use the generated video_id)
    video_id_from_path = Path(video_path).parent.name  # This is the video_id directory
    
    config = {
        "session": {
            "video_id": video_id_from_path,
            "output_dir": output_dir,
        },
        "steps": [
            # Ingestion - use disk mode so pipeline finds the video in data/raw/
            {
                "name": "ingestion",
                "module": "video_loader",
                "enabled": True,
                "input_source": "disk",
                "save_output": False,
                "params": {}
            },
            # YOLO COCO Detection
            {
                "name": "yolo_coco_detection",
                "module": "yolo_detector",
                "enabled": True,
                "save_output": False,
                "params": {
                    "model_path": "models/pretrained/yolov8s-seg.pt",
                    "task": "segment",
                    "conf_threshold": 0.25,
                    "source_name": "yolo_coco",
                    "progress_every": 30
                }
            },
            # Detection Filter
            {
                "name": "detection_filter",
                "module": "detection_filter",
                "enabled": True,
                "input_from_step": "yolo_coco_detection",
                "save_output": False,
                "params": {
                    "min_confidence": 0.05,
                    "largest_selector": {
                        "enabled": True,
                        "classes": ["person"]
                    }
                }
            },
            # Tracking
            {
                "name": "disc_tracking",
                "module": "model_tracker",
                "enabled": True,
                "input_from_step": "detection_filter",
                "save_output": False,
                "params": {
                    "enabled": True,
                    "classes_to_track": ["frisbee", "person", "sports ball"],
                    "single_object_classes": ["frisbee", "sports ball"],
                    "min_det_score": 0.05,
                    "high_det_score": 0.15,
                    "max_age_frames": 30,
                    "min_hits_to_confirm": 2,
                    "association": {
                        "iou_weight": 0.5,
                        "center_weight": 0.5,
                        "max_center_dist_px": 200,
                        "min_iou": 0.01
                    }
                }
            },
            # Track Refiner
            {
                "name": "track_refiner",
                "module": "track_refiner",
                "enabled": True,
                "input_from_step": "disc_tracking",
                "save_output": False,
                "params": {
                    "enabled": True,
                    "classes_to_refine": ["frisbee", "person", "sports ball"],
                    "smoothing": {
                        "enabled": True,
                        "method": "moving_average",
                        "window": 5
                    }
                }
            },
            # Metrics Calculator
            {
                "name": "metrics_calculator",
                "module": "metrics_calculator",
                "enabled": True,
                "input_from_step": "track_refiner",
                "save_output": True,
                "params": {
                    "target_class": "frisbee",  # Will try frisbee, sports ball, etc.
                    "fallback_classes": ["sports ball", "person"],
                    "physical_params": {
                        "disc_diameter_m": 0.45,
                        "disc_weight_kg": 20.0,
                        "bar_weight_kg": 20.0,
                        "num_discs": 2
                    }
                }
            }
        ]
    }
    
    # Add initial selection if provided
    if selection_data:
        # Add size filter to detection_filter step
        for step in config["steps"]:
            if step["name"] == "detection_filter":
                step["params"]["size_filter"] = {
                    "enabled": True,
                    "reference_radius": selection_data.get("radius", 50),
                    "tolerance": 0.30,
                    "classes": ["frisbee", "sports ball"]
                }
                step["params"]["initial_selector"] = {
                    "enabled": True,
                    "class_name": "frisbee",
                    "reference_center": selection_data.get("center"),
                    "reference_radius": selection_data.get("radius")
                }
    
    return config


def build_api_results(
    video_id: str,
    video_path: str,
    tracked_objects: list,
    metrics_data: list,
    video_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build API response from pipeline outputs.
    
    Args:
        video_id: Video identifier
        video_path: Path to the video
        tracked_objects: List of TrackedObject from tracking
        metrics_data: List of metric dictionaries from JSON
        video_metadata: Video metadata dict
        
    Returns:
        Results dictionary for API response
    """
    # Use video metadata from runner (already extracted)
    metadata = {
        "fps": video_metadata.get("fps", 30.0),
        "width": video_metadata.get("width", 0),
        "height": video_metadata.get("height", 0),
        "duration_s": video_metadata.get("duration_s", 0.0),
        "total_frames": video_metadata.get("total_frames", 0)
    }
    
    # Convert tracks
    tracks = []
    if tracked_objects:
        # Group detections by track_id
        track_dict: Dict[int, Dict] = {}
        
        for obj in tracked_objects:
            if not hasattr(obj, 'track_id') or obj.track_id is None:
                continue
                
            tid = obj.track_id
            if tid not in track_dict:
                class_name = "unknown"
                if hasattr(obj, 'detection') and hasattr(obj.detection, 'class_name'):
                    class_name = obj.detection.class_name
                elif hasattr(obj, 'class_name'):
                    class_name = obj.class_name
                    
                track_dict[tid] = {
                    "track_id": tid,
                    "class_name": class_name,
                    "frames": {},
                    "trajectory": []
                }
            
            # Get frame index
            frame_idx = getattr(obj, 'frame_idx', len(track_dict[tid]["frames"]))
            
            # Extract bbox
            det = obj.detection if hasattr(obj, 'detection') else obj
            bbox = det.bbox if hasattr(det, 'bbox') else [0, 0, 0, 0]
            
            track_dict[tid]["frames"][str(frame_idx)] = {
                "bbox": {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3])
                },
                "confidence": float(det.confidence) if hasattr(det, 'confidence') else 0.5
            }
            
            # Add mask if available (as polygon points)
            if hasattr(det, 'mask') and det.mask:
                # Mask might be a numpy array or list of points
                mask_data = det.mask
                if hasattr(mask_data, 'tolist'):
                    mask_data = mask_data.tolist()
                track_dict[tid]["frames"][str(frame_idx)]["mask"] = mask_data
            
            # Add to trajectory from history
            if hasattr(obj, 'history') and obj.history:
                track_dict[tid]["trajectory"] = [
                    [float(p[0]), float(p[1])] for p in obj.history
                ]
        
        tracks = list(track_dict.values())
    
    # Convert metrics from list of dicts
    metrics = {
        "frames": [],
        "time_s": [],
        "x_m": [],
        "y_m": [],
        "height_m": [],
        "vx_m_s": [],
        "vy_m_s": [],
        "speed_m_s": [],
        "accel_m_s2": [],
        "kinetic_energy_j": [],
        "potential_energy_j": [],
        "total_energy_j": [],
        "power_w": []
    }
    
    summary = {
        "peak_speed_m_s": 0.0,
        "peak_power_w": 0.0,
        "max_height_m": 0.0,
        "min_height_m": 0.0,
        "lift_duration_s": metadata["duration_s"],
        "total_frames": metadata["total_frames"]
    }
    
    if metrics_data:
        # Extract columns from list of dicts
        for row in metrics_data:
            metrics["frames"].append(row.get("frame_idx", 0))
            metrics["time_s"].append(row.get("time_s", 0.0))
            metrics["x_m"].append(row.get("x_m", 0.0))
            metrics["y_m"].append(row.get("y_m", 0.0))
            metrics["height_m"].append(row.get("height_m", 0.0))
            metrics["vx_m_s"].append(row.get("vx_m_s", 0.0))
            metrics["vy_m_s"].append(row.get("vy_m_s", 0.0))
            metrics["speed_m_s"].append(row.get("speed_m_s", 0.0))
            metrics["accel_m_s2"].append(row.get("accel_m_s2", 0.0))
            metrics["kinetic_energy_j"].append(row.get("kinetic_energy_j", 0.0))
            metrics["potential_energy_j"].append(row.get("potential_energy_j", 0.0))
            metrics["total_energy_j"].append(row.get("total_energy_j", 0.0))
            metrics["power_w"].append(row.get("power_w", 0.0))
        
        # Calculate summary
        if metrics["speed_m_s"]:
            summary["peak_speed_m_s"] = max(metrics["speed_m_s"])
        if metrics["power_w"]:
            summary["peak_power_w"] = max(abs(p) for p in metrics["power_w"])
        if metrics["height_m"]:
            summary["max_height_m"] = max(metrics["height_m"])
            summary["min_height_m"] = min(metrics["height_m"])
    
    return {
        "video_id": video_id,
        "status": "completed",
        "metadata": metadata,
        "tracks": tracks,
        "metrics": metrics,
        "summary": summary,
        "processed_at": datetime.now().isoformat()
    }


async def process_video_task(video_id: str):
    """
    Background task to process a video.
    
    This runs the pipeline and updates job status.
    """
    storage = get_storage()
    job = storage.get_job(video_id)
    
    if job is None:
        print(f"[Task] Job not found: {video_id}")
        return
    
    try:
        # Update status to processing
        storage.update_job(
            video_id,
            status=ProcessingStatus.PROCESSING,
            progress=0.0,
            message="Starting pipeline..."
        )
        
        video_path = job.video_path
        results_dir = storage.get_results_dir(video_id)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlink in data/raw/ where pipeline expects videos
        # The pipeline looks for: data/raw/{video_id}.mp4
        import os
        raw_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Use video_id as the filename
        expected_video_path = raw_dir / f"{video_id}.mp4"
        
        # Create symlink if it doesn't exist
        if not expected_video_path.exists():
            try:
                os.symlink(video_path, expected_video_path)
            except OSError:
                # If symlink fails, copy the file
                import shutil
                shutil.copy2(video_path, expected_video_path)
        
        # Import pipeline components
        # We do this inside the function to avoid circular imports
        import sys
        
        # Add src to path if needed
        src_path = Path(__file__).parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from pipeline.runner import PipelineRunner
        from pipeline.config import PipelineConfig
        
        # Create pipeline config
        config_dict = create_api_pipeline_config(
            video_path=video_path,
            output_dir=str(results_dir)
        )
        
        # Write config to YAML file
        config_yaml_path = results_dir / "pipeline_config.yaml"
        with open(config_yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # Update status
        storage.update_job(
            video_id,
            progress=0.1,
            current_step="initializing",
            message="Initializing pipeline..."
        )
        
        # Import and register modules
        from input_layer.video_loader import VideoLoader
        from perception.yolo_detector import YoloDetector
        from analysis.detection_filter import DetectionFilter
        from analysis.model_tracker import ModelTracker
        from analysis.track_refiner import TrackRefiner
        from analysis.metrics_calculator import MetricsCalculator
        
        # Create pipeline runner
        runner = PipelineRunner(config_yaml_path)
        
        # Register modules
        runner.register_step("video_loader", VideoLoader)
        runner.register_step("yolo_detector", YoloDetector)
        runner.register_step("detection_filter", DetectionFilter)
        runner.register_step("model_tracker", ModelTracker)
        runner.register_step("track_refiner", TrackRefiner)
        runner.register_step("metrics_calculator", MetricsCalculator)
        
        # Update status
        storage.update_job(
            video_id,
            progress=0.2,
            current_step="running_pipeline",
            message="Running AI analysis..."
        )
        
        # Run pipeline (synchronously in thread)
        await asyncio.to_thread(runner.run)
        
        # Update progress
        storage.update_job(
            video_id,
            progress=0.9,
            current_step="extracting_results",
            message="Extracting results..."
        )
        
        # Get video metadata from runner
        video_metadata = {
            "fps": 30.0,
            "width": 0,
            "height": 0,
            "duration_s": 0.0,
            "total_frames": 0
        }
        if hasattr(runner, 'video_metadata') and runner.video_metadata:
            vm = runner.video_metadata
            video_metadata = {
                "fps": getattr(vm, 'fps', 30.0),
                "width": getattr(vm, 'width', 0),
                "height": getattr(vm, 'height', 0),
                "duration_s": getattr(vm, 'duration_seconds', 0.0),
                "total_frames": getattr(vm, 'total_frames', 0)
            }
        
        # Load metrics from saved JSON (more reliable than from memory)
        metrics_data = []
        metrics_file = results_dir / "metrics_calculator_output.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics_data = json.load(f)
        
        # Get tracks from pipeline outputs
        tracked_objects = runner.step_outputs.get("track_refiner", [])
        
        # Build results
        results = build_api_results(
            video_id=video_id,
            video_path=video_path,
            tracked_objects=tracked_objects,
            metrics_data=metrics_data,
            video_metadata=video_metadata
        )
        
        # Save results
        results_path = storage.save_results(video_id, results)
        
        # Update job as completed
        storage.update_job(
            video_id,
            status=ProcessingStatus.COMPLETED,
            progress=1.0,
            current_step=None,
            message="Processing completed successfully",
            results_path=str(results_path)
        )
        
        print(f"[Task] Completed processing: {video_id}")
        
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"[Task] Error processing {video_id}: {error_msg}")
        print(traceback_str)
        
        storage.update_job(
            video_id,
            status=ProcessingStatus.FAILED,
            progress=0.0,
            message=f"Processing failed: {error_msg}",
            error=traceback_str
        )


# Simple task queue using asyncio
_task_queue: asyncio.Queue = None
_worker_task: asyncio.Task = None


async def _worker():
    """Background worker that processes tasks from the queue."""
    global _task_queue
    while True:
        video_id = await _task_queue.get()
        try:
            await process_video_task(video_id)
        except Exception as e:
            print(f"[Worker] Error: {e}")
        finally:
            _task_queue.task_done()


def start_worker():
    """Start the background worker."""
    global _task_queue, _worker_task
    if _task_queue is None:
        _task_queue = asyncio.Queue()
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker())


async def enqueue_processing(video_id: str):
    """Add a video to the processing queue."""
    global _task_queue
    if _task_queue is None:
        _task_queue = asyncio.Queue()
        start_worker()
    await _task_queue.put(video_id)
