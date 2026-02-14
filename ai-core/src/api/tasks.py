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
from .storage import get_storage


# Default tracking backend (server-side config, can be changed via API)
DEFAULT_TRACKING_BACKEND = "cutie"

# Human-readable step names for progress reporting to mobile clients
STEP_DISPLAY_NAMES = {
    "ingestion": "Loading video",
    "cutie_disc_tracking": "Tracking disc",
    "yolo_person_detection": "Detecting athlete",
    "yolo_pose": "Estimating pose",
    "merged_detections": "Merging detections",
    "yolo_coco_detection": "Detecting objects",
    "detection_filter": "Filtering detections",
    "disc_tracking": "Assigning tracks",
    "track_refiner": "Refining trajectory",
    "metrics_calculator": "Calculating metrics",
}

# Server-side runtime config (mutable, set by control panel)
_server_config = {
    "tracking_backend": DEFAULT_TRACKING_BACKEND,
    "enable_person_detection": False,
    "enable_pose_estimation": False,
}


def get_server_tracking_backend() -> str:
    """Get the current server-side tracking backend."""
    return _server_config.get("tracking_backend", DEFAULT_TRACKING_BACKEND)


def get_server_model_config() -> dict:
    """Get the full server-side model configuration."""
    return dict(_server_config)


def set_server_tracking_backend(backend: str):
    """Set the server-side tracking backend."""
    _server_config["tracking_backend"] = backend
    print(f"[Config] Server tracking backend set to: {backend.upper()}")


def set_server_model_config(config: dict):
    """Set the full server-side model configuration."""
    if "tracking_backend" in config:
        _server_config["tracking_backend"] = config["tracking_backend"]
    if "enable_person_detection" in config:
        _server_config["enable_person_detection"] = bool(config["enable_person_detection"])
    if "enable_pose_estimation" in config:
        _server_config["enable_pose_estimation"] = bool(config["enable_pose_estimation"])
    print(f"[Config] Model config updated: backend={_server_config['tracking_backend'].upper()}, "
          f"person={'ON' if _server_config['enable_person_detection'] else 'OFF'}, "
          f"pose={'ON' if _server_config['enable_pose_estimation'] else 'OFF'}")



def create_api_pipeline_config(
    video_path: str,
    output_dir: str,
    selection_data: Optional[Dict[str, Any]] = None,
    tracking_backend: Optional[str] = None,
    enable_person_detection: bool = False,
    enable_pose_estimation: bool = False
) -> Dict[str, Any]:
    """
    Create a pipeline configuration for API processing.
    
    Args:
        video_path: Path to the uploaded video
        output_dir: Directory for pipeline outputs
        selection_data: Optional disc selection {center: [x,y], radius: r}
        tracking_backend: "cutie" or "yolo" (defaults to DEFAULT_TRACKING_BACKEND)
        enable_person_detection: Include YOLO person segmentation
        enable_pose_estimation: Include YOLO pose estimation
    
    Returns:
        Pipeline configuration dictionary
    """
    backend = tracking_backend or DEFAULT_TRACKING_BACKEND
    
    if backend == "cutie":
        return _create_cutie_pipeline_config(
            video_path, output_dir, selection_data,
            enable_person_detection, enable_pose_estimation
        )
    else:
        return _create_yolo_pipeline_config(
            video_path, output_dir, selection_data,
            enable_person_detection, enable_pose_estimation
        )


def _create_cutie_pipeline_config(
    video_path: str,
    output_dir: str,
    selection_data: Optional[Dict[str, Any]] = None,
    enable_person_detection: bool = False,
    enable_pose_estimation: bool = False
) -> Dict[str, Any]:
    """
    Create pipeline config using Cutie for disc tracking.
    
    Minimal flow (disc only): ingestion → cutie_disc → tracker → refiner → metrics
    With person: adds yolo_person + merger step
    With pose: adds yolo_pose step
    """
    video_id_from_path = Path(video_path).parent.name
    
    # Determine which classes the tracker handles
    classes = ["frisbee"]
    if enable_person_detection:
        classes.append("person")
    
    steps = [
        # Stage 1: Video Loading
        {
            "name": "ingestion",
            "module": "video_loader",
            "enabled": True,
            "input_source": "disk",
            "save_output": False,
            "params": {}
        },
        # Stage 2: Cutie Disc Tracking (always enabled)
        {
            "name": "cutie_disc_tracking",
            "module": "cutie_tracker",
            "enabled": True,
            "input_source": "memory",
            "input_from_step": "ingestion",
            "save_output": True,
            "params": {
                "weights_path": "models/pretrained/cutie/cutie-base-mega.pth",
                "max_internal_size": 480,
                "mem_every": 5,
                "target_class_name": "frisbee",
                "min_mask_area": 100,
                "progress_every": 30
            }
        },
    ]
    
    # Determine what feeds into the tracker
    tracker_input = "cutie_disc_tracking"
    
    # Optionally add YOLO person detection + merger
    if enable_person_detection:
        steps.append({
            "name": "yolo_person_detection",
            "module": "yolo_detector",
            "enabled": True,
            "input_source": "memory",
            "input_from_step": "ingestion",
            "save_output": False,
            "params": {
                "model_path": "models/pretrained/yolov8s-seg.pt",
                "task": "segment",
                "conf_threshold": 0.25,
                "source_name": "yolo_coco",
                "progress_every": 30
            }
        })
        steps.append({
            "name": "merged_detections",
            "module": "detection_merger",
            "enabled": True,
            "input_source": "memory",
            "input_from_step": ["cutie_disc_tracking", "yolo_person_detection"],
            "save_output": False,
            "params": {}
        })
        tracker_input = "merged_detections"
    
    # Optionally add YOLO pose estimation
    if enable_pose_estimation:
        steps.append({
            "name": "yolo_pose",
            "module": "yolo_detector",
            "enabled": True,
            "input_source": "memory",
            "input_from_step": "ingestion",
            "save_output": True,
            "params": {
                "model_path": "models/pretrained/yolov8n-pose.pt",
                "task": "pose",
                "conf_threshold": 0.3,
                "source_name": "yolo_pose",
                "progress_every": 30
            }
        })
    
    # Tracker, refiner, metrics (always)
    steps.extend([
        {
            "name": "disc_tracking",
            "module": "model_tracker",
            "enabled": True,
            "input_source": "memory",
            "input_from_step": tracker_input,
            "save_output": False,
            "params": {
                "enabled": True,
                "classes_to_track": classes,
                "single_object_classes": classes,
                "min_det_score": 0.05,
                "high_det_score": 0.15,
                "max_age_frames": 90,
                "min_hits_to_confirm": 1,
                "association": {
                    "max_center_dist_px": 300,
                },
                "progress_every": 30
            }
        },
        {
            "name": "track_refiner",
            "module": "track_refiner",
            "enabled": True,
            "input_from_step": "disc_tracking",
            "save_output": False,
            "params": {
                "enabled": True,
                "classes_to_refine": classes,
                "smoothing": {
                    "enabled": True,
                    "method": "savgol",
                    "window": 11
                }
            }
        },
        {
            "name": "metrics_calculator",
            "module": "metrics_calculator",
            "enabled": True,
            "input_from_step": "track_refiner",
            "save_output": True,
            "params": {
                "target_class": "frisbee",
                "fallback_classes": ["person"] if enable_person_detection else [],
                "smooth_window": 11,
                "physical_params": {
                    "disc_diameter_m": 0.45,
                    "disc_weight_kg": 20.0,
                    "bar_weight_kg": 20.0,
                    "num_discs": 2
                }
            }
        }
    ])
    
    config = {
        "session": {
            "video_id": video_id_from_path,
            "output_dir": output_dir,
        },
        "steps": steps
    }
    
    # Add disc selection data
    if selection_data:
        for step in config["steps"]:
            if step["name"] == "cutie_disc_tracking":
                step["params"]["initial_selection"] = {
                    "center": selection_data.get("center"),
                    "radius": selection_data.get("radius")
                }
            if step["name"] == "disc_tracking":
                step["params"]["initial_selection"] = {
                    "class_name": "frisbee",
                    "center": selection_data.get("center"),
                    "radius": selection_data.get("radius")
                }
    
    return config


def _create_yolo_pipeline_config(
    video_path: str,
    output_dir: str,
    selection_data: Optional[Dict[str, Any]] = None,
    enable_person_detection: bool = False,
    enable_pose_estimation: bool = False
) -> Dict[str, Any]:
    """
    Create pipeline config using YOLO for all detection + tracking (original flow).
    
    Note: In YOLO mode, person detection is inherent (COCO detects all classes).
    The enable_person_detection flag controls whether person tracks are kept.
    """
    video_id_from_path = Path(video_path).parent.name
    
    config = {
        "session": {
            "video_id": video_id_from_path,
            "output_dir": output_dir,
        },
        "steps": [
            # Ingestion
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
                    "conf_threshold": 0.10,
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
                        "method": "savgol",
                        "window": 11
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
                    "target_class": "frisbee",
                    "fallback_classes": ["sports ball", "person"],
                    "smooth_window": 11,
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
        for step in config["steps"]:
            if step["name"] == "detection_filter":
                step["params"]["size_filter"] = {
                    "enabled": False,
                    "reference_radius": selection_data.get("radius", 50),
                    "tolerance": 0.50,
                    "classes": ["frisbee", "sports ball"]
                }
                step["params"]["initial_selector"] = {
                    "enabled": True,
                    "class_name": "frisbee",
                    "reference_center": selection_data.get("center"),
                    "reference_radius": selection_data.get("radius")
                }
            if step["name"] == "disc_tracking":
                step["params"]["initial_selection"] = {
                    "class_name": "frisbee",
                    "center": selection_data.get("center"),
                    "radius": selection_data.get("radius")
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
    
    # Convert tracks from Dict[frame_idx, List[TrackedObject]]
    tracks = []
    if tracked_objects:
        # Group detections by track_id
        track_dict: Dict[int, Dict] = {}
        
        # tracked_objects is Dict[frame_idx, List[TrackedObject]]
        if isinstance(tracked_objects, dict):
            for frame_idx, frame_objects in tracked_objects.items():
                if not isinstance(frame_objects, list):
                    continue
                    
                for obj in frame_objects:
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
                    
                    # Extract detection data
                    det = obj.detection if hasattr(obj, 'detection') else obj
                    confidence = float(det.confidence) if hasattr(det, 'confidence') else 0.5
                    
                    frame_data: Dict[str, Any] = {
                        "confidence": confidence,
                        "bbox": None,
                        "mask": None
                    }
                    
                    # Add bbox if available
                    bbox = det.bbox if hasattr(det, 'bbox') else None
                    if bbox is not None and len(bbox) == 4:
                        frame_data["bbox"] = {
                            "x1": float(bbox[0]),
                            "y1": float(bbox[1]),
                            "x2": float(bbox[2]),
                            "y2": float(bbox[3])
                        }
                    
                    # Add mask if available (as polygon points)
                    if hasattr(det, 'mask') and det.mask is not None:
                        mask_data = det.mask
                        if hasattr(mask_data, 'tolist'):
                            mask_data = mask_data.tolist()
                        frame_data["mask"] = mask_data
                    
                    track_dict[tid]["frames"][str(frame_idx)] = frame_data
        
        # Build trajectory from mask centroids (preferred) or bbox centers,
        # then smooth with Savitzky-Golay to remove frame-to-frame jitter
        import numpy as np
        for tid, track_data in track_dict.items():
            sorted_frame_indices = sorted(track_data["frames"].keys(), key=lambda k: int(k))
            raw_points = []
            for fidx in sorted_frame_indices:
                fdata = track_data["frames"][fidx]
                cx, cy = None, None
                
                # Prefer mask centroid for trajectory
                if fdata.get("mask") and isinstance(fdata["mask"], list) and len(fdata["mask"]) > 0:
                    pts = np.array(fdata["mask"])
                    cx = float(pts[:, 0].mean())
                    cy = float(pts[:, 1].mean())
                elif fdata.get("bbox"):
                    bbox = fdata["bbox"]
                    cx = (bbox["x1"] + bbox["x2"]) / 2.0
                    cy = (bbox["y1"] + bbox["y2"]) / 2.0
                
                if cx is not None and cy is not None:
                    raw_points.append([cx, cy])
            
            # Apply Savitzky-Golay smoothing to trajectory
            if len(raw_points) >= 7:
                from scipy.signal import savgol_filter
                arr = np.array(raw_points)
                # window_length must be odd and <= len; polyorder=3 preserves peaks
                window = min(11, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
                arr[:, 0] = savgol_filter(arr[:, 0], window_length=window, polyorder=3)
                arr[:, 1] = savgol_filter(arr[:, 1], window_length=window, polyorder=3)
                track_data["trajectory"] = [[float(x), float(y)] for x, y in arr]
            else:
                track_data["trajectory"] = raw_points
        
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
            message="Starting analysis..."
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
        
        # Get selection data and model config from job + server config
        selection_data = job.selection_data
        # Priority: job-specific > server config > hardcoded default
        tracking_backend = job.tracking_backend or get_server_tracking_backend()
        model_config = get_server_model_config()
        enable_person = model_config.get("enable_person_detection", False)
        enable_pose = model_config.get("enable_pose_estimation", False)
        
        if selection_data:
            print(f"[Task] Using disc selection: center={selection_data['center']}, radius={selection_data['radius']}")
        print(f"[Task] *** Tracking backend: {tracking_backend.upper()} | "
              f"Person: {'ON' if enable_person else 'OFF'} | "
              f"Pose: {'ON' if enable_pose else 'OFF'} ***")
        
        # Store the actual backend used in the job (so UI shows correct info)
        storage.update_job(video_id, tracking_backend=tracking_backend)
        
        # Create pipeline config
        config_dict = create_api_pipeline_config(
            video_path=video_path,
            output_dir=str(results_dir),
            selection_data=selection_data,
            tracking_backend=tracking_backend,
            enable_person_detection=enable_person,
            enable_pose_estimation=enable_pose
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
            message="Initializing models..."
        )
        
        # Import and register modules
        from input_layer.video_loader import VideoLoader
        from perception.yolo_detector import YoloDetector
        from perception.cutie_tracker import CutieTracker
        from analysis.detection_filter import DetectionFilter
        from analysis.merger import DetectionMerger
        from analysis.model_tracker import ModelTracker
        from analysis.track_refiner import TrackRefiner
        from analysis.metrics_calculator import MetricsCalculator
        
        # Create pipeline runner
        runner = PipelineRunner(config_yaml_path)
        
        # Register all available modules (pipeline config determines which are used)
        runner.register_step("video_loader", VideoLoader)
        runner.register_step("yolo_detector", YoloDetector)
        runner.register_step("cutie_tracker", CutieTracker)
        runner.register_step("detection_filter", DetectionFilter)
        runner.register_step("detection_merger", DetectionMerger)
        runner.register_step("model_tracker", ModelTracker)
        runner.register_step("track_refiner", TrackRefiner)
        runner.register_step("metrics_calculator", MetricsCalculator)
        
        # Set up progress callback so each pipeline step updates the job
        def on_step_progress(step_name, step_idx, total_steps):
            # Map step progress linearly from 0.15 to 0.85
            progress = 0.15 + (step_idx / max(1, total_steps)) * 0.70
            display_name = STEP_DISPLAY_NAMES.get(step_name, step_name)
            storage.update_job(
                video_id,
                progress=round(progress, 2),
                current_step=step_name,
                message=f"{display_name} ({step_idx + 1}/{total_steps})"
            )
        
        runner.on_step_start = on_step_progress
        
        # Update status
        storage.update_job(
            video_id,
            progress=0.15,
            current_step="running_pipeline",
            message="Processing video..."
        )
        
        # Run pipeline (synchronously in thread)
        await asyncio.to_thread(runner.run)
        
        # Update progress
        storage.update_job(
            video_id,
            progress=0.9,
            current_step="extracting_results",
            message="Building results..."
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
            message="Analysis complete",
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
