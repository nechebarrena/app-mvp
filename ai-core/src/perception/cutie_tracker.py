"""
Cutie-based Video Object Segmentation tracker.

Uses the Cutie model (CVPR 2024) for semi-supervised video object segmentation.
Given an initial mask derived from disc selection (center + radius), tracks the
object through all video frames producing per-frame Detection objects.

This module is designed as a drop-in alternative to YoloDetector for disc tracking.
It outputs Dict[int, List[Detection]] — the same format as YoloDetector — so all
downstream modules (track_refiner, metrics_calculator) work without changes.

Key differences from YOLO-based detection:
  - Does NOT rely on COCO class labels (no "frisbee"/"sports ball" confusion)
  - Requires an initial mask/selection (prompt-based, not class-based)
  - Produces continuous tracking across ALL frames (no detection gaps)
  - Outputs high-quality segmentation masks

Design note: This module is structured for future portability to mobile.
The core logic (mask→bbox conversion, confidence estimation) is kept separate
from the Cutie model loading, so it can be swapped for a lighter model later.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from torchvision.transforms.functional import to_tensor

from domain.ports import IPipelineStep
from domain.entities import VideoSession, Detection


def _get_device() -> torch.device:
    """Select the best available device (MPS for Mac, CUDA for Linux/Windows, CPU fallback)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_circular_mask(height: int, width: int, center: Tuple[float, float],
                          radius: float) -> np.ndarray:
    """Create a binary circular mask from disc selection parameters.
    
    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        center: (x, y) center of the disc selection.
        radius: Radius of the disc selection in pixels.
    
    Returns:
        Binary mask as uint8 numpy array (0=background, 1=object).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(round(radius))
    cv2.circle(mask, (cx, cy), r, 1, -1)
    return mask


def _mask_to_detection(binary_mask: np.ndarray, frame_idx: int,
                       confidence: float, class_name: str) -> Optional[Detection]:
    """Convert a binary segmentation mask to a Detection entity.
    
    This function extracts bbox, contour polygon, center, and radius from
    the binary mask. It's designed to be portable — the same logic will be
    needed on mobile when processing model outputs.
    
    Args:
        binary_mask: Binary mask (H, W) where 1=object.
        frame_idx: Frame index (for debug/logging).
        confidence: Confidence score for this detection.
        class_name: Class name to assign (e.g. "frisbee").
    
    Returns:
        Detection object or None if the mask is empty.
    """
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    
    # Bounding box
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    
    # Contour polygon (largest contour only for clean output)
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask_polygon = None
    if contours:
        # Use the largest contour
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 3:
            mask_polygon = largest.squeeze().tolist()
    
    # Estimate radius from mask area (assuming roughly circular object)
    area = len(xs)
    radius_px = float(np.sqrt(area / np.pi))
    
    # Circularity score: how close to a perfect circle
    if contours:
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            shape_score = min(1.0, circularity)
        else:
            shape_score = 0.0
    else:
        shape_score = 0.0
    
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
        mask=mask_polygon,
        source="cutie",
        radius_px=radius_px,
        shape_score=shape_score,
    )


class CutieTracker(IPipelineStep[VideoSession, Dict[int, List[Detection]]]):
    """
    Video object segmentation tracker using Cutie model.
    
    Takes a VideoSession and disc selection, produces per-frame detections
    with segmentation masks for the tracked disc object.
    
    Config params:
        weights_path: Path to Cutie model weights (default: models/pretrained/cutie/cutie-base-mega.pth)
        max_internal_size: Max short-edge for internal processing (default: 480)
        mem_every: Store memory every N frames (default: 5)
        target_class_name: Class name for output detections (default: "frisbee")
        min_mask_area: Minimum mask area in pixels to consider valid (default: 100)
        progress_every: Print progress every N frames (default: 30)
    """

    def run(self, input_data: VideoSession, config: Dict[str, Any]) -> Dict[int, List[Detection]]:
        video_path = Path(input_data.file_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # --- Config ---
        weights_path = config.get("weights_path", "models/pretrained/cutie/cutie-base-mega.pth")
        max_internal_size = int(config.get("max_internal_size", 480))
        mem_every = int(config.get("mem_every", 5))
        target_class_name = config.get("target_class_name", "frisbee")
        min_mask_area = int(config.get("min_mask_area", 100))
        progress_every = max(1, int(config.get("progress_every", 30)))

        # --- Get disc selection ---
        selection_data = self._get_selection(config)
        if selection_data is None:
            raise ValueError(
                "[CutieTracker] No disc selection data found. "
                "Cutie requires an initial mask from disc selection."
            )
        
        center = selection_data["center"]
        radius = selection_data["radius"]
        print(f"[CutieTracker] Disc selection: center=({center[0]:.0f}, {center[1]:.0f}), radius={radius:.0f}")

        # --- Load model ---
        device = _get_device()
        print(f"[CutieTracker] Loading Cutie model on {device}...")
        
        cutie_model, processor = self._load_model(weights_path, device, max_internal_size, mem_every)
        print(f"[CutieTracker] Model loaded successfully")

        # --- Open video ---
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[CutieTracker] Video: {video_path.name}, {total_frames} frames @ {fps_video:.1f} fps, {frame_w}x{frame_h}")

        # --- Create initial mask ---
        initial_mask = _create_circular_mask(frame_h, frame_w, center, radius)
        initial_mask_tensor = torch.from_numpy(initial_mask).long().to(device)
        
        mask_area = initial_mask.sum()
        print(f"[CutieTracker] Initial mask area: {mask_area} pixels")

        # --- Process all frames ---
        results_map: Dict[int, List[Detection]] = {}
        frame_idx = 0
        start_ts = time.time()
        frames_with_detection = 0

        with torch.inference_mode():
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # Convert BGR -> RGB -> tensor
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image_tensor = to_tensor(frame_rgb).float().to(device)

                # Run Cutie
                if frame_idx == 0:
                    output_prob = processor.step(
                        image_tensor, initial_mask_tensor, objects=[1]
                    )
                else:
                    output_prob = processor.step(image_tensor)

                # Convert to binary mask
                output_mask = processor.output_prob_to_mask(output_prob)
                mask_np = output_mask.cpu().numpy().astype(np.uint8)

                # Derive confidence from output probabilities
                # output_prob shape: (num_objects+1, H, W), channel 0=bg, channel 1=object
                if output_prob.shape[0] > 1:
                    obj_prob = output_prob[1]  # object channel
                    mask_pixels = (mask_np == 1)
                    if mask_pixels.any():
                        confidence = float(obj_prob[mask_pixels].mean().cpu())
                    else:
                        confidence = 0.0
                else:
                    confidence = 1.0 if (mask_np == 1).any() else 0.0

                # Convert mask to Detection
                binary_mask = (mask_np == 1).astype(np.uint8)
                mask_area_frame = binary_mask.sum()
                
                if mask_area_frame >= min_mask_area:
                    detection = _mask_to_detection(
                        binary_mask, frame_idx, confidence, target_class_name
                    )
                    if detection is not None:
                        results_map[frame_idx] = [detection]
                        frames_with_detection += 1

                frame_idx += 1

                # Progress reporting
                if frame_idx == 1 or frame_idx % progress_every == 0 or frame_idx == total_frames:
                    elapsed = time.time() - start_ts
                    fps_proc = frame_idx / max(1e-6, elapsed)
                    if total_frames > 0:
                        pct = 100.0 * frame_idx / total_frames
                        remaining = total_frames - frame_idx
                        eta_s = remaining / max(1e-6, fps_proc)
                        print(
                            f"[CutieTracker] {frame_idx}/{total_frames} "
                            f"({pct:.1f}%) | {fps_proc:.1f} fps | ETA {eta_s:.0f}s | "
                            f"dets={frames_with_detection}",
                            end="\r"
                        )

        cap.release()

        # Cleanup GPU memory
        del processor, cutie_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed_total = time.time() - start_ts
        fps_avg = frame_idx / max(1e-6, elapsed_total)

        print(f"\n[CutieTracker] Completed {frame_idx} frames in {elapsed_total:.1f}s "
              f"({fps_avg:.1f} fps avg)")
        print(f"[CutieTracker] Detections: {frames_with_detection}/{frame_idx} frames "
              f"({frames_with_detection/max(1, frame_idx)*100:.1f}% coverage)")

        return results_map

    def _get_selection(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract disc selection data from config or selection file.
        
        Supports multiple sources (in order of priority):
          1. Direct 'initial_selection' in config params
          2. Selection file path in '_selection_file' (injected by pipeline runner)
          3. Selection file in 'selection_file' param
        """
        # Direct config
        if "initial_selection" in config:
            sel = config["initial_selection"]
            if "center" in sel and "radius" in sel:
                return sel

        # Selection file (injected by pipeline runner or passed as param)
        selection_file = config.get("_selection_file") or config.get("selection_file")
        if selection_file:
            sel_path = Path(selection_file)
            if sel_path.exists():
                with open(sel_path) as f:
                    data = json.load(f)
                if "center" in data and "radius" in data:
                    return data

        return None

    def _load_model(self, weights_path: str, device: torch.device,
                    max_internal_size: int, mem_every: int):
        """Load and configure the Cutie model.
        
        Returns:
            Tuple of (cutie_model, inference_processor)
        """
        # Add vendor path for cutie imports
        vendor_cutie_path = str(Path(__file__).parent.parent.parent / "vendors" / "cutie")
        if vendor_cutie_path not in sys.path:
            sys.path.insert(0, vendor_cutie_path)
        
        from omegaconf import OmegaConf, open_dict
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        from cutie.model.cutie import CUTIE
        from cutie.inference.inference_core import InferenceCore
        
        # Clear any previous hydra state
        GlobalHydra.instance().clear()
        
        # Load Cutie config via hydra
        config_dir = os.path.join(vendor_cutie_path, "cutie", "config")
        with initialize_config_dir(version_base='1.3.2', config_dir=config_dir):
            cfg = compose(config_name="eval_config")
        
        # Override config for our use case
        with open_dict(cfg):
            cfg['weights'] = weights_path
            cfg['max_internal_size'] = max_internal_size
            cfg['mem_every'] = mem_every
            cfg['flip_aug'] = False
            cfg['save_aux'] = False
            cfg['chunk_size'] = -1
            cfg['stagger_updates'] = min(5, mem_every)
        
        # Load model
        cutie = CUTIE(cfg).to(device).eval()
        model_weights = torch.load(weights_path, map_location=device, weights_only=False)
        cutie.load_weights(model_weights)
        
        # Create inference processor
        processor = InferenceCore(cutie, cfg=cfg)
        processor.max_internal_size = max_internal_size
        
        return cutie, processor

    def save_result(self, data: Dict[int, List[Detection]], output_path: Path) -> None:
        """Serialize detection map to JSON."""
        serializable_data = {
            str(k): [d.model_dump() for d in v]
            for k, v in data.items()
        }
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    def load_result(self, input_path: Path) -> Dict[int, List[Detection]]:
        """Deserialize detection map from JSON."""
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
        return {
            int(k): [Detection(**d) for d in v]
            for k, v in raw_data.items()
        }
