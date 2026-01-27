"""
Multi-Model Video Renderer

Creates a multi-panel video for comparing outputs from different detection models.
Each panel shows detections from one model with:
- Frame number in top-left
- Model name/source in header
- Legend showing all classes that appear in the video (visible from frame 0)
- Color-coded bounding boxes, masks, and keypoints

Panel layout adapts to number of models: 1, 2, 3, or 2x2 grid for 4.
"""

import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from collections import defaultdict

from domain.ports import IPipelineStep
from domain.entities import Detection
from domain.label_mapper import LabelMapper


# Distinctive color palette (BGR format)
COLOR_PALETTE = [
    (255, 100, 100),   # Light blue
    (100, 255, 100),   # Light green
    (100, 100, 255),   # Light red
    (255, 255, 100),   # Cyan
    (255, 100, 255),   # Magenta
    (100, 255, 255),   # Yellow
    (200, 150, 100),   # Steel blue
    (100, 200, 150),   # Sea green
    (150, 100, 200),   # Purple
    (200, 200, 100),   # Teal
    (100, 200, 200),   # Gold
    (200, 100, 200),   # Pink
]

# COCO keypoint skeleton
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face
    (5, 6),                               # Shoulders
    (5, 7), (7, 9),                       # Left Arm
    (6, 8), (8, 10),                      # Right Arm
    (5, 11), (6, 12),                     # Torso
    (11, 12),                             # Hips
    (11, 13), (13, 15),                   # Left Leg
    (12, 14), (14, 16)                    # Right Leg
]


class MultiModelRenderer(IPipelineStep[Any, Path]):
    """
    Renders a multi-panel comparison video from multiple model outputs.
    
    Input: Either:
        - List of Dict[int, List[Detection]] (from multiple steps)
        - Dict mapping source_name -> {frame_idx -> [Detection, ...]}
    
    The source_name for each model is extracted from the Detection.source field.
    
    Config params:
        video_source: Path to original video
        output_filename: Path for output video
        panel_width: Width per panel in pixels (default: 640)
        font_scale: Base font scale (default: 0.5)
        show_confidence: Show confidence scores (default: True)
        show_class_id: Show class IDs (default: False)
        progress_every: Print progress every N frames (default: 20)
        
        # Mask rendering options
        mask_alpha: Opacity of mask fill, 0.0-1.0 (default: 0.45)
        mask_contour: Draw mask contour/border (default: True)
        mask_contour_thickness: Thickness of mask contour (default: 2)
        
        # Label mapping (optional) - unify labels across models
        label_mapping: Dict mapping global_label -> {source: model_label}
        visualize_labels: List of global labels to show (filters output)
        use_global_labels: Display global labels instead of model labels (default: True when mapping exists)
    """

    def _get_class_color(self, class_name: str, class_colors: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Get consistent color for a class name."""
        if class_name not in class_colors:
            idx = len(class_colors) % len(COLOR_PALETTE)
            class_colors[class_name] = COLOR_PALETTE[idx]
        return class_colors[class_name]

    def _scan_all_classes(self, detections: Dict[int, List[Detection]]) -> Set[str]:
        """Scan all frames to find unique class names."""
        classes = set()
        for frame_dets in detections.values():
            for det in frame_dets:
                classes.add(det.class_name)
        return classes

    def _draw_detection(
        self,
        frame: np.ndarray,
        det: Detection,
        color: Tuple[int, int, int],
        font_scale: float,
        thickness: int,
        show_conf: bool,
        show_id: bool,
        mask_alpha: float = 0.45,
        mask_contour: bool = True,
        mask_contour_thickness: int = 2,
    ) -> None:
        """Draw a single detection on the frame."""
        # Draw mask if available
        if det.mask:
            pts = np.array(det.mask, dtype=np.int32).reshape((-1, 1, 2))
            # Fill mask with semi-transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, mask_alpha, frame, 1.0 - mask_alpha, 0, frame)
            # Draw contour for better visibility
            if mask_contour:
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=mask_contour_thickness)

        # Draw keypoints if available (pose)
        if det.keypoints:
            kpts = det.keypoints
            for kpt in kpts:
                x, y, conf = kpt
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
            
            for i, j in COCO_SKELETON:
                if i < len(kpts) and j < len(kpts):
                    pt1, pt2 = kpts[i], kpts[j]
                    if pt1[2] > 0.3 and pt2[2] > 0.3:
                        cv2.line(
                            frame,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            color,
                            thickness,
                        )

        # Draw bounding box
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        parts = [det.class_name]
        if show_id:
            parts.append(f"id:{det.class_id}")
        if show_conf:
            parts.append(f"{det.confidence:.2f}")
        label = " ".join(parts)

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    def _draw_legend(
        self,
        frame: np.ndarray,
        classes: List[str],
        class_colors: Dict[str, Tuple[int, int, int]],
        font_scale: float,
        start_y: int,
    ) -> None:
        """Draw class legend on the frame."""
        y = start_y
        for cls_name in sorted(classes):
            color = class_colors.get(cls_name, (255, 255, 255))
            # Draw color box
            cv2.rectangle(frame, (10, y - 12), (25, y + 2), color, -1)
            cv2.rectangle(frame, (10, y - 12), (25, y + 2), (255, 255, 255), 1)
            # Draw class name
            cv2.putText(
                frame,
                cls_name,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.8,
                (255, 255, 255),
                1,
            )
            y += 18

    def _normalize_input(
        self,
        input_data: Union[List[Dict[int, List[Detection]]], Dict[str, Dict[int, List[Detection]]]],
    ) -> Dict[str, Dict[int, List[Detection]]]:
        """
        Normalize input to Dict[source_name -> {frame_idx -> [Detection]}].
        
        Handles both:
        - List of detection dicts (extracts source from Detection.source)
        - Already normalized dict
        """
        if isinstance(input_data, dict) and all(isinstance(k, str) for k in input_data.keys()):
            # Already normalized
            return input_data
        
        if isinstance(input_data, list):
            # List of detection dicts from multiple steps
            result: Dict[str, Dict[int, List[Detection]]] = {}
            
            for det_map in input_data:
                if not isinstance(det_map, dict):
                    continue
                    
                for frame_idx, detections in det_map.items():
                    for det in detections:
                        source = det.source or "unknown"
                        if source not in result:
                            result[source] = {}
                        if frame_idx not in result[source]:
                            result[source][frame_idx] = []
                        result[source][frame_idx].append(det)
            
            return result
        
        raise ValueError(f"Unsupported input type: {type(input_data)}")

    def run(
        self,
        input_data: Union[List[Dict[int, List[Detection]]], Dict[str, Dict[int, List[Detection]]]],
        config: Dict[str, Any],
    ) -> Path:
        video_source = config.get("video_source")
        if not video_source:
            raise ValueError("MultiModelRenderer requires 'video_source' in params.")

        source_path = Path(video_source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source video not found: {source_path}")

        output_name = config.get("output_filename", "comparison.mp4")
        output_path = Path(output_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        panel_width = int(config.get("panel_width", 640))
        font_scale = float(config.get("font_scale", 0.5))
        show_conf = bool(config.get("show_confidence", True))
        show_id = bool(config.get("show_class_id", False))
        progress_every = max(1, int(config.get("progress_every", 20)))
        
        # Mask rendering options
        mask_alpha = float(config.get("mask_alpha", 0.45))
        mask_contour = bool(config.get("mask_contour", True))
        mask_contour_thickness = int(config.get("mask_contour_thickness", 2))

        # Label mapping configuration
        label_mapper = LabelMapper.from_config(config)
        use_global_labels = bool(config.get("use_global_labels", label_mapper.is_configured))
        
        if label_mapper.is_configured:
            print(f"[MultiModelRenderer] {label_mapper.describe()}")

        # Normalize input format
        normalized_data = self._normalize_input(input_data)
        
        # Get source names (models)
        source_names = list(normalized_data.keys())
        n_models = len(source_names)

        if n_models == 0:
            raise ValueError("No model outputs provided to MultiModelRenderer")

        print(f"[MultiModelRenderer] Rendering {n_models} models: {source_names}")

        # Pre-scan all classes per model for legends (respecting label mapping)
        model_classes: Dict[str, Set[str]] = {}
        class_colors: Dict[str, Tuple[int, int, int]] = {}

        for source in source_names:
            # Scan raw classes
            raw_classes = self._scan_all_classes(normalized_data[source])
            
            if label_mapper.is_configured:
                # Filter and translate to global labels
                mapped_classes: Set[str] = set()
                for cls in raw_classes:
                    include, global_label = label_mapper.should_include(source, cls)
                    if include and global_label:
                        mapped_classes.add(global_label)
                    elif include and not global_label:
                        # Pass-through mode
                        mapped_classes.add(cls)
                model_classes[source] = mapped_classes
            else:
                model_classes[source] = raw_classes
            
            # Assign colors for all classes (use global labels for consistency)
            for cls in model_classes[source]:
                self._get_class_color(cls, class_colors)

        # Open video
        cap = cv2.VideoCapture(str(source_path))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Calculate panel dimensions maintaining aspect ratio
        aspect = orig_h / orig_w
        panel_h = int(panel_width * aspect)

        # Layout: 1=single, 2=side-by-side, 3=row, 4=2x2
        if n_models <= 3:
            grid_w, grid_h = n_models, 1
        else:
            grid_w, grid_h = 2, (n_models + 1) // 2

        canvas_w = panel_width * grid_w
        canvas_h = panel_h * grid_h

        print(f"[MultiModelRenderer] Output: {canvas_w}x{canvas_h}, {n_models} panels @ {panel_width}x{panel_h}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))
        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")

        frame_idx = 0
        start_ts = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            for i, source in enumerate(source_names):
                # Panel position
                px = (i % grid_w) * panel_width
                py = (i // grid_w) * panel_h

                # Resize frame for this panel
                panel = cv2.resize(frame.copy(), (panel_width, panel_h))

                # Scale factor for coordinates
                sx = panel_width / orig_w
                sy = panel_h / orig_h

                # Draw detections (with label mapping/filtering)
                dets = normalized_data[source].get(frame_idx, [])
                for det in dets:
                    # Check if this detection should be included
                    include, global_label = label_mapper.should_include(source, det.class_name)
                    if not include:
                        continue
                    
                    # Determine display label
                    display_label = global_label if (use_global_labels and global_label) else det.class_name
                    
                    # Scale bbox and create detection with display label
                    x1, y1, x2, y2 = det.bbox
                    scaled_det = Detection(
                        class_id=det.class_id,
                        class_name=display_label,  # Use mapped label for display
                        confidence=det.confidence,
                        bbox=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                        mask=[[p[0] * sx, p[1] * sy] for p in det.mask] if det.mask else None,
                        keypoints=[[k[0] * sx, k[1] * sy, k[2]] for k in det.keypoints] if det.keypoints else None,
                        source=det.source,
                    )
                    # Use display label for color consistency across models
                    color = self._get_class_color(display_label, class_colors)
                    self._draw_detection(
                        panel, scaled_det, color, font_scale, 2, show_conf, show_id,
                        mask_alpha=mask_alpha,
                        mask_contour=mask_contour,
                        mask_contour_thickness=mask_contour_thickness,
                    )

                # Draw header
                header_h = 30
                cv2.rectangle(panel, (0, 0), (panel_width, header_h), (40, 40, 40), -1)
                cv2.putText(
                    panel,
                    f"{source}",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 1.2,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    panel,
                    f"frame: {frame_idx}",
                    (panel_width - 120, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.9,
                    (200, 200, 200),
                    1,
                )

                # Draw legend (classes that appear in this model's output)
                legend_classes = list(model_classes[source])
                if legend_classes:
                    self._draw_legend(panel, legend_classes, class_colors, font_scale, header_h + 20)

                # Place panel on canvas
                canvas[py:py + panel_h, px:px + panel_width] = panel

            out.write(canvas)
            frame_idx += 1

            # Progress
            if frame_idx == 1 or frame_idx % progress_every == 0:
                elapsed = time.time() - start_ts
                fps_proc = frame_idx / max(1e-6, elapsed)
                if total_frames > 0:
                    pct = 100.0 * frame_idx / total_frames
                    eta = (total_frames - frame_idx) / max(1e-6, fps_proc)
                    print(
                        f"[MultiModelRenderer] {frame_idx}/{total_frames} ({pct:.1f}%) | "
                        f"{fps_proc:.1f} fps | ETA {eta:.0f}s",
                        end="\r"
                    )
                else:
                    print(f"[MultiModelRenderer] {frame_idx} frames | {fps_proc:.1f} fps", end="\r")

        cap.release()
        out.release()

        elapsed = time.time() - start_ts
        print(f"\n[MultiModelRenderer] Completed {frame_idx} frames in {elapsed:.1f}s")
        print(f"[MultiModelRenderer] Output saved to: {output_path}")

        return output_path

    def save_result(self, data: Path, output_path: Path) -> None:
        with open(output_path, 'w') as f:
            f.write(str(data))

    def load_result(self, input_path: Path) -> Path:
        with open(input_path, 'r') as f:
            return Path(f.read().strip())
