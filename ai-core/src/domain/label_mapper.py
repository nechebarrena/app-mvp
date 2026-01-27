"""
Label Mapper - Unified label system across multiple models.

Provides:
- Mapping from model-specific labels to global/unified labels
- Filtering detections by global labels
- Reverse lookup (global label -> model-specific label)

Example config:
    label_mapping:
      disco:
        yolo_custom: "discos"
        yolo_coco: "frisbee"
      atleta:
        yolo_custom: "atleta"
        yolo_coco: "person"
        yolo_pose: "person"
      barra:
        yolo_custom: "barra"
        # yolo_coco: not mapped (this concept doesn't exist in COCO)
    
    visualize_labels: ["disco", "atleta"]
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass


@dataclass
class LabelMapping:
    """Stores the mapping configuration."""
    # global_label -> { source: model_label }
    global_to_model: Dict[str, Dict[str, str]]
    # (source, model_label) -> global_label (reverse lookup)
    model_to_global: Dict[Tuple[str, str], str]
    # Set of global labels to visualize/process
    active_labels: Set[str]


class LabelMapper:
    """
    Handles label translation between model-specific and global labels.
    
    Usage:
        mapper = LabelMapper.from_config(config)
        global_label = mapper.to_global("yolo_custom", "discos")  # -> "disco"
        model_label = mapper.to_model("disco", "yolo_coco")  # -> "frisbee"
        is_active = mapper.is_active("disco")  # -> True
    """
    
    def __init__(self, mapping: LabelMapping):
        self.mapping = mapping
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LabelMapper":
        """
        Create LabelMapper from config dict.
        
        Expected config structure:
            {
                "label_mapping": {
                    "global_label": {
                        "source1": "model_label1",
                        "source2": "model_label2",
                    },
                    ...
                },
                "visualize_labels": ["label1", "label2", ...]  # optional
            }
        """
        raw_mapping = config.get("label_mapping", {})
        visualize = config.get("visualize_labels", None)
        
        global_to_model: Dict[str, Dict[str, str]] = {}
        model_to_global: Dict[Tuple[str, str], str] = {}
        
        for global_label, source_map in raw_mapping.items():
            if not isinstance(source_map, dict):
                continue
            global_to_model[global_label] = dict(source_map)
            for source, model_label in source_map.items():
                # Store reverse mapping (source, model_label) -> global_label
                key = (str(source), str(model_label))
                model_to_global[key] = str(global_label)
        
        # Determine active labels
        if visualize is not None:
            active_labels = set(str(v) for v in visualize)
        else:
            # If not specified, all mapped labels are active
            active_labels = set(global_to_model.keys())
        
        return cls(LabelMapping(
            global_to_model=global_to_model,
            model_to_global=model_to_global,
            active_labels=active_labels,
        ))
    
    @classmethod
    def identity(cls) -> "LabelMapper":
        """Create an identity mapper (pass-through, no filtering)."""
        return cls(LabelMapping(
            global_to_model={},
            model_to_global={},
            active_labels=set(),
        ))
    
    @property
    def is_configured(self) -> bool:
        """Check if any mappings are defined."""
        return len(self.mapping.global_to_model) > 0
    
    def to_global(self, source: str, model_label: str) -> Optional[str]:
        """
        Translate model-specific label to global label.
        
        Returns None if no mapping exists for this (source, label) pair.
        """
        key = (source, model_label)
        return self.mapping.model_to_global.get(key, None)
    
    def to_model(self, global_label: str, source: str) -> Optional[str]:
        """
        Translate global label to model-specific label for a given source.
        
        Returns None if this global label is not mapped for this source.
        """
        source_map = self.mapping.global_to_model.get(global_label, {})
        return source_map.get(source, None)
    
    def is_active(self, global_label: str) -> bool:
        """Check if a global label is in the active/visualize list."""
        if not self.mapping.active_labels:
            return True  # No filter = all active
        return global_label in self.mapping.active_labels
    
    def should_include(self, source: str, model_label: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a detection should be included and get its global label.
        
        Returns:
            (include: bool, global_label: Optional[str])
            
        If no mapping is configured, returns (True, None) to pass through.
        """
        if not self.is_configured:
            # No mapping configured - pass through everything
            return (True, None)
        
        global_label = self.to_global(source, model_label)
        if global_label is None:
            # No mapping for this label - exclude
            return (False, None)
        
        if not self.is_active(global_label):
            # Mapped but not in active list - exclude
            return (False, global_label)
        
        # Include with global label
        return (True, global_label)
    
    def get_sources_for_label(self, global_label: str) -> List[str]:
        """Get list of sources that have a mapping for this global label."""
        source_map = self.mapping.global_to_model.get(global_label, {})
        return list(source_map.keys())
    
    def get_all_global_labels(self) -> List[str]:
        """Get all defined global labels."""
        return list(self.mapping.global_to_model.keys())
    
    def get_active_labels(self) -> List[str]:
        """Get list of active/visualize labels."""
        return list(self.mapping.active_labels)
    
    def describe(self) -> str:
        """Return a human-readable description of the mapping."""
        if not self.is_configured:
            return "LabelMapper: No mapping configured (pass-through mode)"
        
        lines = ["LabelMapper Configuration:"]
        lines.append(f"  Active labels: {sorted(self.mapping.active_labels)}")
        lines.append("  Mappings:")
        for global_label, source_map in sorted(self.mapping.global_to_model.items()):
            active_marker = "✓" if self.is_active(global_label) else " "
            sources_str = ", ".join(f"{s}:{l}" for s, l in sorted(source_map.items()))
            lines.append(f"    [{active_marker}] {global_label}: {sources_str}")
        return "\n".join(lines)
