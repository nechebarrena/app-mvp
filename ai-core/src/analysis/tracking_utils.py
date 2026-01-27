import numpy as np
from typing import Tuple, List, Optional, Any

class KalmanFilter2D:
    """
    A simple 2D Kalman Filter for constant velocity model (x, y, vx, vy).
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    def __init__(self, x: float, y: float, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        # State Vector [x, y, vx, vy]
        self.x = np.array([x, y, 0., 0.], dtype=float)
        
        # State Covariance
        self.P = np.eye(4) * 10.0
        
        # State Transition Matrix (F)
        # x_next = x + vx*dt
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement Matrix (H)
        # We measure x, y
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process Noise Covariance (Q)
        self.Q = np.eye(4) * process_noise
        
        # Measurement Noise Covariance (R)
        self.R = np.eye(2) * measurement_noise
        
        # Identity
        self.I = np.eye(4)

    def predict(self) -> Tuple[float, float]:
        """Predicts the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, z_meas: Tuple[float, float]):
        """Updates the state with a new measurement."""
        z = np.array(z_meas)
        y = z - self.H @ self.x # Residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])
        
    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


class GeometryUtils:
    @staticmethod
    def get_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def get_area(bbox: Tuple[float, float, float, float]) -> float:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    @staticmethod
    def get_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if max(w, h) == 0: return 0.0
        return min(w, h) / max(w, h) # Always <= 1.0 (Squareness)

    @staticmethod
    def get_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return interArea / float(box1Area + box2Area - interArea + 1e-6)
