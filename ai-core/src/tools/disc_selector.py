"""
Disc Selector: Interactive tool for manually selecting disc center and radius.

Usage:
    result = DiscSelector.select_from_video("path/to/video.mp4")
    # Returns: {"center": (cx, cy), "radius": r, "accepted": True/False}

The tool displays the first frame of the video and allows the user to:
1. Click "Centro" button, then click on the frame to mark the disc center
2. Click "Borde" button, then click on the frame to mark the disc edge
3. The disc is drawn with transparency based on center and edge
4. Click "Aceptar" to confirm or "Resetear" to start over
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class Button:
    """Simple button representation."""
    x: int
    y: int
    w: int
    h: int
    label: str
    enabled: bool = True
    active: bool = False  # Currently selected mode
    
    def contains(self, px: int, py: int) -> bool:
        """Check if point is inside button."""
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h
    
    def draw(self, img: np.ndarray) -> None:
        """Draw button on image."""
        # Colors
        if not self.enabled:
            bg_color = (80, 80, 80)
            text_color = (150, 150, 150)
        elif self.active:
            bg_color = (50, 150, 50)  # Green when active
            text_color = (255, 255, 255)
        else:
            bg_color = (60, 60, 60)
            text_color = (255, 255, 255)
        
        # Background
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg_color, -1)
        # Border
        border_color = (255, 255, 255) if self.enabled else (100, 100, 100)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), border_color, 2)
        
        # Text (centered)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(self.label, font, font_scale, thickness)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(img, self.label, (tx, ty), font, font_scale, text_color, thickness)


class DiscSelector:
    """Interactive disc selection tool using OpenCV."""
    
    WINDOW_NAME = "Disc Selector - Select center and edge"
    
    def __init__(self, frame: np.ndarray):
        self.original_frame = frame.copy()
        self.frame_h, self.frame_w = frame.shape[:2]
        
        # Selection state
        self.center: Optional[Tuple[int, int]] = None
        self.edge: Optional[Tuple[int, int]] = None
        self.radius: Optional[float] = None
        
        # Mode: None, "center", "edge"
        self.mode: Optional[str] = None
        
        # Result
        self.accepted = False
        self.cancelled = False
        
        # UI Layout
        self.panel_height = 60
        self.canvas_h = self.frame_h + self.panel_height
        self.canvas_w = self.frame_w
        
        # Buttons
        btn_w = 100
        btn_h = 40
        btn_y = self.frame_h + 10
        spacing = 20
        
        # Calculate button positions (centered)
        total_btns_w = 4 * btn_w + 3 * spacing
        start_x = (self.canvas_w - total_btns_w) // 2
        
        self.btn_center = Button(start_x, btn_y, btn_w, btn_h, "Centro")
        self.btn_edge = Button(start_x + btn_w + spacing, btn_y, btn_w, btn_h, "Borde", enabled=False)
        self.btn_accept = Button(start_x + 2 * (btn_w + spacing), btn_y, btn_w, btn_h, "Aceptar", enabled=False)
        self.btn_reset = Button(start_x + 3 * (btn_w + spacing), btn_y, btn_w, btn_h, "Resetear")
        
        self.buttons = [self.btn_center, self.btn_edge, self.btn_accept, self.btn_reset]
    
    def _draw_cross(self, img: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int], size: int = 15) -> None:
        """Draw a cross marker at the given point."""
        x, y = point
        cv2.line(img, (x - size, y), (x + size, y), color, 2)
        cv2.line(img, (x, y - size), (x, y + size), color, 2)
    
    def _draw_disc(self, img: np.ndarray) -> None:
        """Draw the selected disc with transparency."""
        if self.center is None or self.radius is None:
            return
        
        cx, cy = self.center
        r = int(self.radius)
        
        # Create overlay for transparency
        overlay = img.copy()
        
        # Draw filled circle
        cv2.circle(overlay, (cx, cy), r, (100, 255, 100), -1)
        
        # Blend with original
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Draw circle border
        cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
    
    def _render(self) -> np.ndarray:
        """Render the current state."""
        # Create canvas
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        
        # Copy frame
        frame_display = self.original_frame.copy()
        
        # Draw disc if we have center and radius
        if self.center and self.radius:
            self._draw_disc(frame_display)
        
        # Draw center cross
        if self.center:
            self._draw_cross(frame_display, self.center, (0, 0, 255))  # Red
        
        # Draw edge cross
        if self.edge:
            self._draw_cross(frame_display, self.edge, (255, 0, 0))  # Blue
        
        # Place frame on canvas
        canvas[:self.frame_h, :self.frame_w] = frame_display
        
        # Draw panel background
        canvas[self.frame_h:, :] = (40, 40, 40)
        
        # Draw buttons
        for btn in self.buttons:
            btn.draw(canvas)
        
        # Draw instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        instructions = ""
        if self.mode == "center":
            instructions = "Click en el CENTRO del disco"
        elif self.mode == "edge":
            instructions = "Click en el BORDE del disco"
        elif self.center and self.radius:
            instructions = "Presiona Aceptar o Resetear"
        else:
            instructions = "Selecciona Centro para comenzar"
        
        cv2.putText(canvas, instructions, (10, self.frame_h + self.panel_height - 10),
                   font, 0.5, (200, 200, 200), 1)
        
        return canvas
    
    def _update_button_states(self) -> None:
        """Update button enabled/active states based on current selection."""
        # Edge button enabled only if center is set
        self.btn_edge.enabled = self.center is not None
        
        # Accept button enabled only if both center and edge are set
        self.btn_accept.enabled = self.center is not None and self.edge is not None
        
        # Update active states based on mode
        self.btn_center.active = self.mode == "center"
        self.btn_edge.active = self.mode == "edge"
    
    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is on a button
            for btn in self.buttons:
                if btn.contains(x, y) and btn.enabled:
                    self._handle_button_click(btn)
                    return
            
            # Click on frame area
            if y < self.frame_h:
                self._handle_frame_click(x, y)
    
    def _handle_button_click(self, btn: Button) -> None:
        """Handle button click."""
        if btn == self.btn_center:
            self.mode = "center"
        elif btn == self.btn_edge:
            self.mode = "edge"
        elif btn == self.btn_accept:
            self.accepted = True
        elif btn == self.btn_reset:
            self._reset()
        
        self._update_button_states()
    
    def _handle_frame_click(self, x: int, y: int) -> None:
        """Handle click on the frame."""
        if self.mode == "center":
            self.center = (x, y)
            # If edge was set, recalculate radius
            if self.edge:
                self.radius = np.sqrt((self.edge[0] - x)**2 + (self.edge[1] - y)**2)
        elif self.mode == "edge":
            self.edge = (x, y)
            if self.center:
                self.radius = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        
        self._update_button_states()
    
    def _reset(self) -> None:
        """Reset selection."""
        self.center = None
        self.edge = None
        self.radius = None
        self.mode = None
        self._update_button_states()
    
    def run(self) -> Dict[str, Any]:
        """Run the selector and return the result."""
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)
        
        print("[DiscSelector] Window opened. Select center and edge of the disc.")
        print("[DiscSelector] Press ESC to cancel, or use the buttons.")
        
        while True:
            canvas = self._render()
            cv2.imshow(self.WINDOW_NAME, canvas)
            
            key = cv2.waitKey(30) & 0xFF
            
            # ESC to cancel
            if key == 27:
                self.cancelled = True
                break
            
            # Check if accepted
            if self.accepted:
                break
        
        cv2.destroyWindow(self.WINDOW_NAME)
        
        if self.accepted and self.center and self.radius:
            result = {
                "center": self.center,
                "radius": float(self.radius),
                "edge": self.edge,
                "accepted": True,
            }
            print(f"[DiscSelector] Selection accepted: center={self.center}, radius={self.radius:.1f}")
            return result
        else:
            print("[DiscSelector] Selection cancelled.")
            return {"accepted": False}
    
    @classmethod
    def select_from_video(cls, video_path: str, frame_index: int = 0) -> Dict[str, Any]:
        """
        Open video and run selection on the specified frame.
        
        Args:
            video_path: Path to video file
            frame_index: Which frame to use for selection (default: first frame)
            
        Returns:
            Dict with keys: center, radius, edge, accepted
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Seek to frame
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Cannot read frame {frame_index} from video")
        
        # Run selector
        selector = cls(frame)
        return selector.run()
    
    @classmethod
    def select_from_frame(cls, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run selection on a provided frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            Dict with keys: center, radius, edge, accepted
        """
        selector = cls(frame)
        return selector.run()


def save_selection(result: Dict[str, Any], output_path: str) -> None:
    """Save selection result to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tuples to lists for JSON serialization
    data = {
        "center": list(result["center"]) if result.get("center") else None,
        "radius": result.get("radius"),
        "edge": list(result["edge"]) if result.get("edge") else None,
        "accepted": result.get("accepted", False),
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[DiscSelector] Selection saved to: {output_path}")


def load_selection(input_path: str) -> Dict[str, Any]:
    """Load selection result from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to tuples
    if data.get("center"):
        data["center"] = tuple(data["center"])
    if data.get("edge"):
        data["edge"] = tuple(data["edge"])
    
    return data


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python disc_selector.py <video_path> [output_json]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "disc_selection.json"
    
    result = DiscSelector.select_from_video(video_path)
    
    if result["accepted"]:
        save_selection(result, output_path)
    else:
        print("Selection cancelled, no output saved.")
