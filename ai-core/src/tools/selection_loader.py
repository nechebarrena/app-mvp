"""
Selection Loader: Pipeline step to load manual disc selection.

This step loads a pre-saved disc selection (center + radius) from a JSON file
and makes it available to downstream steps for filtering/focusing tracking.

Usage in YAML config:
    - name: disc_selection
      module: "selection_loader"
      params:
        selection_file: "path/to/disc_selection.json"
        # Or provide inline:
        # center: [320, 240]
        # radius: 50

The output is a dict with:
    - center: (x, y) tuple
    - radius: float
    - bbox: (x1, y1, x2, y2) bounding box derived from circle
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from domain.ports import IPipelineStep
from domain.entities import VideoSession


class SelectionLoader(IPipelineStep[VideoSession, Dict[str, Any]]):
    """
    Loads disc selection from file or config and outputs selection data.
    """
    
    def run(self, input_data: VideoSession, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load selection from file or inline config.
        
        Config params:
            selection_file: Path to JSON file with selection
            center: [x, y] - inline center (alternative to file)
            radius: float - inline radius (alternative to file)
        """
        selection_file = config.get("selection_file")
        
        if selection_file:
            # Load from file
            selection_path = Path(selection_file)
            if not selection_path.is_absolute():
                # Relative to workspace
                selection_path = Path(config.get("_workspace_root", ".")) / selection_file
            
            if not selection_path.exists():
                raise FileNotFoundError(f"Selection file not found: {selection_path}")
            
            with open(selection_path, 'r') as f:
                data = json.load(f)
            
            center = tuple(data["center"]) if data.get("center") else None
            radius = data.get("radius")
            
            print(f"[SelectionLoader] Loaded from file: {selection_path}")
        else:
            # Load from inline config
            center_list = config.get("center")
            radius = config.get("radius")
            
            if center_list is None or radius is None:
                raise ValueError("SelectionLoader requires either 'selection_file' or 'center' + 'radius' in config")
            
            center = tuple(center_list)
            print(f"[SelectionLoader] Using inline config")
        
        if center is None or radius is None:
            raise ValueError("Invalid selection: center or radius is None")
        
        # Compute bounding box from circle
        cx, cy = center
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        
        result = {
            "center": center,
            "radius": float(radius),
            "bbox": bbox,
            "video_path": input_data.file_path,
            "video_width": input_data.specs.width,
            "video_height": input_data.specs.height,
        }
        
        print(f"[SelectionLoader] Selection: center={center}, radius={radius:.1f}")
        print(f"[SelectionLoader] Bounding box: {bbox}")
        
        return result
    
    def save_result(self, data: Dict[str, Any], output_path: Path) -> None:
        """Save selection to JSON."""
        # Convert tuples to lists for JSON
        serializable = {
            "center": list(data["center"]) if data.get("center") else None,
            "radius": data.get("radius"),
            "bbox": list(data["bbox"]) if data.get("bbox") else None,
            "video_path": data.get("video_path"),
            "video_width": data.get("video_width"),
            "video_height": data.get("video_height"),
        }
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def load_result(self, input_path: Path) -> Dict[str, Any]:
        """Load selection from JSON."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to tuples
        if data.get("center"):
            data["center"] = tuple(data["center"])
        if data.get("bbox"):
            data["bbox"] = tuple(data["bbox"])
        
        return data
