"""
Generic YOLO detector that supports detection, segmentation, and pose estimation.

This module provides a unified interface for running any YOLO model with:
- Configurable task type (detect, segment, pose)
- Robust progress reporting
- Consistent output format via Detection entities
"""

import json
import time
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional

from ultralytics import YOLO

from domain.ports import IPipelineStep
from domain.entities import VideoSession, Detection


class YoloDetector(IPipelineStep[VideoSession, Dict[int, List[Detection]]]):
    """
    Generic YOLO detector supporting multiple task types.
    
    Config params:
        model_path: Path to YOLO model weights (.pt file)
        task: "detect" | "segment" | "pose" (default: auto-detect from model)
        conf_threshold: Confidence threshold (default: 0.25)
        source_name: Source identifier for detections (default: derived from model name)
        progress_every: Print progress every N frames (default: 10)
        max_frames: Limit processing to N frames, 0=all (default: 0)
    """

    def run(self, input_data: VideoSession, config: Dict[str, Any]) -> Dict[int, List[Detection]]:
        video_path = Path(input_data.file_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Config
        model_path = config.get("model_path", "yolov8s.pt")
        task = config.get("task", None)  # None = auto-detect
        conf_threshold = float(config.get("conf_threshold", 0.25))
        source_name = config.get("source_name", None)
        progress_every = max(1, int(config.get("progress_every", 10)))
        max_frames = int(config.get("max_frames", 0))

        # Load model
        print(f"[YoloDetector] Loading model: {model_path}")
        model = YOLO(model_path)

        # Auto-detect task from model type if not specified
        if task is None:
            model_type = getattr(model, 'task', None) or 'detect'
            if 'pose' in str(model_path).lower() or model_type == 'pose':
                task = 'pose'
            elif 'seg' in str(model_path).lower() or model_type == 'segment':
                task = 'segment'
            else:
                task = 'detect'

        # Auto-generate source name if not specified
        if source_name is None:
            model_name = Path(model_path).stem  # e.g., "best" or "yolov8s"
            source_name = f"yolo_{model_name}"

        print(f"[YoloDetector] Task: {task}, Source: {source_name}, Conf: {conf_threshold}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        print(f"[YoloDetector] Video: {video_path.name}, {total_frames} frames @ {fps_video:.1f} fps")

        results_map: Dict[int, List[Detection]] = {}
        frame_idx = 0
        start_ts = time.time()
        last_report_ts = start_ts

        while cap.isOpened():
            if max_frames > 0 and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = model.predict(frame, conf=conf_threshold, verbose=False, task=task)

            frame_detections: List[Detection] = []

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                masks = getattr(r, 'masks', None)
                keypoints = getattr(r, 'keypoints', None)

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]

                    # Extract mask if available (segmentation)
                    mask_data = None
                    if masks is not None and len(masks.xy) > i:
                        mask_data = masks.xy[i].tolist()

                    # Extract keypoints if available (pose)
                    kpts_data = None
                    if keypoints is not None and len(keypoints.data) > i:
                        kpts_data = keypoints.data[i].tolist()

                    detection = Detection(
                        class_id=cls,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        mask=mask_data,
                        keypoints=kpts_data,
                        source=source_name,
                    )
                    frame_detections.append(detection)

            if frame_detections:
                results_map[frame_idx] = frame_detections

            frame_idx += 1

            # Progress reporting
            if frame_idx == 1 or frame_idx % progress_every == 0 or frame_idx == total_frames:
                now = time.time()
                elapsed = now - start_ts
                fps_proc = frame_idx / max(1e-6, elapsed)

                if total_frames > 0:
                    pct = 100.0 * frame_idx / total_frames
                    remaining = total_frames - frame_idx
                    eta_s = remaining / max(1e-6, fps_proc)
                    print(
                        f"[YoloDetector] {source_name}: {frame_idx}/{total_frames} "
                        f"({pct:.1f}%) | {fps_proc:.1f} fps | ETA {eta_s:.0f}s | "
                        f"dets={len(frame_detections)}",
                        end="\r"
                    )
                else:
                    print(
                        f"[YoloDetector] {source_name}: {frame_idx} frames | "
                        f"{fps_proc:.1f} fps | dets={len(frame_detections)}",
                        end="\r"
                    )
                last_report_ts = now

        cap.release()

        elapsed_total = time.time() - start_ts
        fps_avg = frame_idx / max(1e-6, elapsed_total)
        total_dets = sum(len(d) for d in results_map.values())
        frames_with_dets = len(results_map)

        print(f"\n[YoloDetector] {source_name}: Completed {frame_idx} frames in {elapsed_total:.1f}s "
              f"({fps_avg:.1f} fps avg)")
        print(f"[YoloDetector] {source_name}: {total_dets} total detections across {frames_with_dets} frames")

        return results_map

    def save_result(self, data: Dict[int, List[Detection]], output_path: Path) -> None:
        serializable_data = {
            str(k): [d.model_dump() for d in v]
            for k, v in data.items()
        }
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    def load_result(self, input_path: Path) -> Dict[int, List[Detection]]:
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
        return {
            int(k): [Detection(**d) for d in v]
            for k, v in raw_data.items()
        }
