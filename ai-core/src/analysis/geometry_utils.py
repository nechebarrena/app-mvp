"""
Geometry utilities for analysis modules.
"""

import numpy as np
from typing import Tuple


class GeometryUtils:
    """Static helper methods for bounding box geometry."""
    
    @staticmethod
    def get_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Get center point of bbox [x1, y1, x2, y2]."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    @staticmethod
    def get_area(bbox: Tuple[float, float, float, float]) -> float:
        """Get area of bbox [x1, y1, x2, y2]."""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    @staticmethod
    def get_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
        """Get aspect ratio (min/max) of bbox. Returns 1.0 for square."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            return 0.0
        return min(w, h) / max(w, h)


class KalmanFilter2D:
    """
    Simple 2D Kalman filter for position tracking with constant velocity model.
    State: [x, y, vx, vy]
    """
    
    def __init__(self, x: float = 0, y: float = 0, 
                 process_noise: float = 1e-3, measurement_noise: float = 1e-1):
        # State [x, y, vx, vy]
        self.state = np.array([x, y, 0.0, 0.0])
        
        # State covariance
        self.P = np.eye(4) * 100.0
        
        # Transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        
        # Process noise
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise
        self.R = np.eye(2) * measurement_noise
    
    def predict(self) -> Tuple[float, float]:
        """Predict next state. Returns (x, y)."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.state[0]), float(self.state[1])
    
    def update(self, measurement: Tuple[float, float]) -> None:
        """Update with measurement (x, y)."""
        z = np.array([measurement[0], measurement[1]])
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    @property
    def position(self) -> Tuple[float, float]:
        return float(self.state[0]), float(self.state[1])
    
    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.state[2]), float(self.state[3])
