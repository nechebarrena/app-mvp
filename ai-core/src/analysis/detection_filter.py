"""
DetectionFilter: Pre-tracking heuristics for filtering detections.

This module applies STATELESS filters to detections BEFORE tracking.
It does NOT use temporal information - each frame is processed independently.

Filters:
- Size filter: Filter detections by expected size (radius/area)
- Confidence filter: Filter by minimum confidence score
- ROI filter: Filter by static region of interest (future)
- Initial selector: Keep only the best detection for a class in frame 0
- Largest selector: Keep only the largest detection per class per frame

Note: Selecting THE correct object among multiple candidates across frames
requires temporal information and is handled by the Tracker (SingleObjectTracker).
The pre-filter's role is to REDUCE candidates, not to SELECT the final one.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from domain.ports import IPipelineStep
from domain.entities import Detection


def box_radius(box: Tuple) -> float:
    """Get approximate radius of box (half of average dimension)."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (w + h) / 4.0


def box_center(box: Tuple) -> Tuple[float, float]:
    """Get center of box [x1, y1, x2, y2]."""
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def box_area(box: Tuple) -> float:
    """Get area of box [x1, y1, x2, y2]."""
    w = max(0, box[2] - box[0])
    h = max(0, box[3] - box[1])
    return w * h


def center_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


@dataclass
class FilterStats:
    """Statistics about filtering for debugging."""
    total_input: int = 0
    passed_confidence: int = 0
    passed_size: int = 0
    passed_roi: int = 0
    final_output: int = 0
    
    rejected_confidence: int = 0
    rejected_size: int = 0
    rejected_roi: int = 0
    rejected_largest: int = 0  # Rejected by largest_selector


class DetectionFilter(IPipelineStep[Dict[int, List[Detection]], Dict[int, List[Detection]]]):
    """
    Pre-tracking detection filter.
    
    Applies stateless filters to reduce candidate detections before tracking.
    
    Config:
        # Confidence filter
        min_confidence: float (default 0.05)
        
        # Size filter (requires reference)
        size_filter:
            enabled: bool (default false)
            reference_radius: float (from selection file or manual)
            tolerance: float (default 0.25 = 25%)
            classes: List[str] (which classes to apply size filter)
            selection_file: str (optional, load reference from file)
        
        # ROI filter (static region, future feature)
        roi_filter:
            enabled: bool (default false)
            x1, y1, x2, y2: int (ROI bounds)
        
        # First-frame selection (select best match to reference in frame 0)
        initial_selector:
            enabled: bool (default false)
            selection_file: str (load center/radius from file)
            class_name: str (which class to select)
            # In frame 0, among valid candidates, pick the one closest to reference
        
        # Largest selector (keep only the largest detection per class per frame)
        # Use case: "only one athlete of interest, probably the largest in frame"
        largest_selector:
            enabled: bool (default false)
            classes: List[str] (which classes to apply)
            # For each frame, keep only the detection with largest bbox area
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.stats_by_class: Dict[str, FilterStats] = {}
        self.reference_center: Optional[Tuple[float, float]] = None
        self.reference_radius: Optional[float] = None
    
    def run(self, input_data: Dict[int, List[Detection]], config: Dict[str, Any]) -> Dict[int, List[Detection]]:
        """
        Apply filters to detections.
        
        Args:
            input_data: Dict[frame_idx, List[Detection]]
            config: Filter configuration
            
        Returns:
            Dict[frame_idx, List[Detection]] - Filtered detections
        """
        self.config = config
        self.stats_by_class = {}
        
        # Load reference if size filter or initial selector is enabled
        self._load_reference(config)
        
        min_confidence = config.get("min_confidence", 0.05)
        size_filter_cfg = config.get("size_filter", {})
        roi_filter_cfg = config.get("roi_filter", {})
        initial_selector_cfg = config.get("initial_selector", {})
        
        size_filter_enabled = size_filter_cfg.get("enabled", False)
        size_tolerance = size_filter_cfg.get("tolerance", 0.25)
        size_classes = set(size_filter_cfg.get("classes", []))
        
        roi_filter_enabled = roi_filter_cfg.get("enabled", False)
        roi_bounds = None
        if roi_filter_enabled:
            roi_bounds = (
                roi_filter_cfg.get("x1", 0),
                roi_filter_cfg.get("y1", 0),
                roi_filter_cfg.get("x2", float('inf')),
                roi_filter_cfg.get("y2", float('inf'))
            )
        
        initial_selector_enabled = initial_selector_cfg.get("enabled", False)
        initial_selector_class = initial_selector_cfg.get("class_name")
        
        # Largest selector config
        largest_selector_cfg = config.get("largest_selector", {})
        largest_selector_enabled = largest_selector_cfg.get("enabled", False)
        largest_selector_classes = set(largest_selector_cfg.get("classes", []))
        
        results: Dict[int, List[Detection]] = {}
        frame_indices = sorted(input_data.keys())
        total_frames = len(frame_indices)
        
        progress_every = config.get("progress_every", 0)
        
        print(f"[DetectionFilter] Processing {total_frames} frames...")
        if size_filter_enabled and self.reference_radius:
            print(f"[DetectionFilter] Size filter: radius={self.reference_radius:.1f}px ±{size_tolerance*100:.0f}%")
            print(f"[DetectionFilter] Size filter applies to classes: {list(size_classes)}")
        if initial_selector_enabled and self.reference_center:
            print(f"[DetectionFilter] Initial selector for '{initial_selector_class}' at {self.reference_center}")
        if largest_selector_enabled:
            print(f"[DetectionFilter] Largest selector for classes: {list(largest_selector_classes)}")
        
        for i, frame_idx in enumerate(frame_indices):
            detections = input_data[frame_idx]
            filtered = []
            
            for det in detections:
                # Initialize stats for this class
                if det.class_name not in self.stats_by_class:
                    self.stats_by_class[det.class_name] = FilterStats()
                stats = self.stats_by_class[det.class_name]
                stats.total_input += 1
                
                # --- Confidence filter ---
                if det.confidence < min_confidence:
                    stats.rejected_confidence += 1
                    continue
                stats.passed_confidence += 1
                
                # --- Size filter (only for specified classes) ---
                if size_filter_enabled and det.class_name in size_classes:
                    if self.reference_radius and self.reference_radius > 0:
                        det_radius = box_radius(det.bbox)
                        size_ratio = det_radius / self.reference_radius
                        
                        min_ratio = 1.0 - size_tolerance
                        max_ratio = 1.0 + size_tolerance
                        
                        if not (min_ratio <= size_ratio <= max_ratio):
                            stats.rejected_size += 1
                            continue
                stats.passed_size += 1
                
                # --- ROI filter ---
                if roi_filter_enabled and roi_bounds:
                    cx, cy = box_center(det.bbox)
                    if not (roi_bounds[0] <= cx <= roi_bounds[2] and 
                            roi_bounds[1] <= cy <= roi_bounds[3]):
                        stats.rejected_roi += 1
                        continue
                stats.passed_roi += 1
                
                # Detection passed all filters
                stats.final_output += 1
                filtered.append(det)
            
            # --- Initial selector (frame 0 only) ---
            if initial_selector_enabled and frame_idx == 0 and self.reference_center:
                # Among filtered detections of the target class, select the closest to reference
                target_class = initial_selector_class
                target_dets = [d for d in filtered if d.class_name == target_class]
                other_dets = [d for d in filtered if d.class_name != target_class]
                
                if target_dets:
                    # Find closest to reference center
                    def dist_to_ref(d):
                        cx, cy = box_center(d.bbox)
                        return center_distance((cx, cy), self.reference_center)
                    
                    target_dets.sort(key=dist_to_ref)
                    best_det = target_dets[0]
                    best_dist = dist_to_ref(best_det)
                    
                    print(f"[DetectionFilter] Frame 0: Selected best '{target_class}' at distance {best_dist:.1f}px from reference")
                    
                    # Keep only the best for target class
                    filtered = other_dets + [best_det]
            
            # --- Largest selector (every frame) ---
            if largest_selector_enabled and largest_selector_classes:
                # For each class in largest_selector_classes, keep only the largest detection
                final_filtered = []
                
                for class_name in largest_selector_classes:
                    class_dets = [d for d in filtered if d.class_name == class_name]
                    if class_dets:
                        # Sort by area (descending) and keep only the largest
                        class_dets.sort(key=lambda d: box_area(d.bbox), reverse=True)
                        largest_det = class_dets[0]
                        final_filtered.append(largest_det)
                        
                        # Update stats for rejected detections
                        rejected_count = len(class_dets) - 1
                        if rejected_count > 0 and class_name in self.stats_by_class:
                            self.stats_by_class[class_name].rejected_largest += rejected_count
                
                # Add detections from classes NOT in largest_selector_classes
                for d in filtered:
                    if d.class_name not in largest_selector_classes:
                        final_filtered.append(d)
                
                filtered = final_filtered
            
            results[frame_idx] = filtered
            
            # Progress
            if progress_every > 0 and (i + 1) % progress_every == 0:
                pct = (i + 1) / total_frames * 100
                print(f"[DetectionFilter] Progress: {i+1}/{total_frames} ({pct:.1f}%)")
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _load_reference(self, config: Dict[str, Any]) -> None:
        """Load reference center and radius from config or file."""
        self.reference_center = None
        self.reference_radius = None
        
        # Try size_filter config
        size_cfg = config.get("size_filter", {})
        if size_cfg.get("reference_radius"):
            self.reference_radius = float(size_cfg["reference_radius"])
        
        # Try initial_selector config
        initial_cfg = config.get("initial_selector", {})
        
        # Load from file - check multiple sources:
        # 1. size_filter.selection_file
        # 2. initial_selector.selection_file
        # 3. _selection_file (injected by pipeline from disc_selection config)
        selection_file = (
            size_cfg.get("selection_file") or 
            initial_cfg.get("selection_file") or 
            config.get("_selection_file")
        )
        
        if selection_file:
            selection_path = Path(selection_file)
            if not selection_path.is_absolute():
                project_root = Path(config.get("_project_root", "."))
                # Try relative to ai-core first
                selection_path = project_root / "ai-core" / selection_file
                if not selection_path.exists():
                    # Try relative to project root
                    selection_path = project_root / selection_file
            
            if selection_path.exists():
                with open(selection_path, 'r') as f:
                    file_data = json.load(f)
                
                if file_data.get("center"):
                    center = file_data["center"]
                    self.reference_center = tuple(center) if isinstance(center, list) else center
                    print(f"[DetectionFilter] Loaded reference center: {self.reference_center}")
                
                if file_data.get("radius"):
                    self.reference_radius = float(file_data["radius"])
                    print(f"[DetectionFilter] Loaded reference radius: {self.reference_radius:.1f}px")
            else:
                print(f"[DetectionFilter] Warning: Selection file not found: {selection_path}")
    
    def _print_summary(self) -> None:
        """Print filtering statistics."""
        print(f"[DetectionFilter] === Filtering Summary ===")
        for class_name, stats in self.stats_by_class.items():
            if stats.total_input > 0:
                pass_rate = stats.final_output / stats.total_input * 100
                print(f"[DetectionFilter] {class_name}: {stats.final_output}/{stats.total_input} passed ({pass_rate:.1f}%)")
                if stats.rejected_confidence > 0:
                    print(f"[DetectionFilter]   - Rejected by confidence: {stats.rejected_confidence}")
                if stats.rejected_size > 0:
                    print(f"[DetectionFilter]   - Rejected by size: {stats.rejected_size}")
                if stats.rejected_roi > 0:
                    print(f"[DetectionFilter]   - Rejected by ROI: {stats.rejected_roi}")
                if stats.rejected_largest > 0:
                    print(f"[DetectionFilter]   - Rejected by largest selector: {stats.rejected_largest}")
    
    def save_result(self, data: Dict[int, List[Detection]], output_path: Path) -> None:
        """Save filtered detections."""
        serializable = {
            str(k): [d.model_dump() for d in v]
            for k, v in data.items()
        }
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def load_result(self, input_path: Path) -> Dict[int, List[Detection]]:
        """Load filtered detections from file."""
        with open(input_path, 'r') as f:
            raw = json.load(f)
        return {
            int(k): [Detection(**d) for d in v]
            for k, v in raw.items()
        }
