import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple

from domain.ports import IPipelineStep
from domain.entities import Detection, TrackedObject

class VideoOverlayRenderer(IPipelineStep[Dict[int, List[Any]], Path]):
    """
    Renders detections (boxes/masks/keypoints) and tracks back onto the original video.
    Input can be Dict[int, List[Detection]] OR Dict[int, List[TrackedObject]].
    """

    COCO_SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Face
        (5, 6),                                   # Shoulders
        (5, 7), (7, 9),                           # Left Arm
        (6, 8), (8, 10),                          # Right Arm
        (5, 11), (6, 12),                         # Torso
        (11, 12),                                 # Hips
        (11, 13), (13, 15),                       # Left Leg
        (12, 14), (14, 16)                        # Right Leg
    ]

    @staticmethod
    def _circularity(area: float, perimeter: float) -> float:
        if perimeter <= 1e-6 or area <= 1e-6:
            return 0.0
        return float((4.0 * np.pi * area) / (perimeter * perimeter + 1e-9))

    @staticmethod
    def _compute_canny_contours_and_circles(
        bgr_frame: np.ndarray,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Returns:
          - edges image (H,W) uint8
          - candidates: list of dicts with keys:
              contour, circle=(x,y,r), score, bbox
        """
        canny1 = int(cfg.get("canny1", 40))
        canny2 = int(cfg.get("canny2", 120))
        blur_ksize = int(cfg.get("blur_ksize", 5))
        use_clahe = bool(cfg.get("use_clahe", True))
        morph_close = bool(cfg.get("morph_close", True))
        morph_close_ksize = int(cfg.get("morph_close_ksize", 3))

        # Detector-aligned filters (match GeomDiscDetector behavior)
        min_area = float(cfg.get("min_area", 800.0))
        min_circularity = float(cfg.get("min_circularity", 0.55))
        min_radius_px = float(cfg.get("min_radius_px", 80.0))
        max_radius_px = float(cfg.get("max_radius_px", 220.0))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        if blur_ksize >= 3 and blur_ksize % 2 == 1:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        if clahe is not None:
            gray = clahe.apply(gray)

        edges = cv2.Canny(gray, threshold1=canny1, threshold2=canny2)
        if morph_close:
            k = max(1, int(morph_close_ksize))
            if k % 2 == 0:
                k += 1
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Dict[str, Any]] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue
            peri = float(cv2.arcLength(cnt, True))
            circ = VideoOverlayRenderer._circularity(area, peri)
            if circ < min_circularity:
                continue
            (xc, yc), r = cv2.minEnclosingCircle(cnt)
            radius = float(r)
            if radius < min_radius_px or radius > max_radius_px:
                continue
            bbox = (float(xc - radius), float(yc - radius), float(xc + radius), float(yc + radius))
            candidates.append(
                {
                    "contour": cnt,
                    "circle": (float(xc), float(yc), float(radius)),
                    "score": float(circ),
                    "bbox": bbox,
                }
            )

        # sort like detector: (confidence=circularity, radius)
        candidates.sort(key=lambda d: (d["score"], d["circle"][2]), reverse=True)
        return edges, candidates

    @staticmethod
    def _compute_canny_edges(bgr_frame: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        canny1 = int(cfg.get("canny1", 40))
        canny2 = int(cfg.get("canny2", 120))
        blur_ksize = int(cfg.get("blur_ksize", 5))
        use_clahe = bool(cfg.get("use_clahe", True))
        morph_close = bool(cfg.get("morph_close", True))
        morph_close_ksize = int(cfg.get("morph_close_ksize", 3))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        if blur_ksize >= 3 and blur_ksize % 2 == 1:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        if clahe is not None:
            gray = clahe.apply(gray)
        edges = cv2.Canny(gray, threshold1=canny1, threshold2=canny2)
        if morph_close:
            k = max(1, int(morph_close_ksize))
            if k % 2 == 0:
                k += 1
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
        return edges

    def run(self, input_data: Dict[int, List[Any]], config: Dict[str, Any]) -> Path:
        video_source = config.get("video_source")
        if not video_source:
             raise ValueError("VideoOverlayRenderer requires 'video_source' in params.")
             
        source_path = Path(video_source)
        if not source_path.exists():
             raise FileNotFoundError(f"Source video not found: {source_path}")

        output_name = config.get("output_filename", "overlay.mp4")
        output_file = Path(output_name)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Optional: write a second, side-by-side debug video showing the Canny pipeline.
        side_by_side_name = config.get("output_side_by_side_filename")
        side_by_side_file = Path(side_by_side_name) if side_by_side_name else None
        if side_by_side_file is not None:
            side_by_side_file.parent.mkdir(parents=True, exist_ok=True)

        # Optional: write a triptych debug video: overlay | canny+contours+circles | yolo raw
        triptych_name = config.get("output_triptych_filename")
        triptych_file = Path(triptych_name) if triptych_name else None
        if triptych_file is not None:
            triptych_file.parent.mkdir(parents=True, exist_ok=True)

        font_scale_mult = float(config.get("font_scale_mult", 1.0))
        font_scale_mult = float(max(0.5, min(4.0, font_scale_mult)))

        canny_cfg = dict(config.get("canny_debug", {}))
        canny_enabled = bool(canny_cfg) and side_by_side_file is not None
        canny1 = int(canny_cfg.get("canny1", 40))
        canny2 = int(canny_cfg.get("canny2", 120))
        blur_ksize = int(canny_cfg.get("blur_ksize", 5))
        use_clahe = bool(canny_cfg.get("use_clahe", True))
        morph_close = bool(canny_cfg.get("morph_close", True))
        morph_close_ksize = int(canny_cfg.get("morph_close_ksize", 3))
        draw_geom = bool(canny_cfg.get("draw_geom_detections", True))
        geom_color = tuple(int(v) for v in canny_cfg.get("geom_color_bgr", [0, 0, 255]))
        invert_edges = bool(canny_cfg.get("invert_edges", False))

        # Triptych uses canny_cfg too (even if side-by-side is disabled)
        triptych_enabled = bool(triptych_file is not None)
        # Candidate drawing colors for panel 2
        best_contour_color = tuple(int(v) for v in canny_cfg.get("best_contour_color_bgr", [0, 255, 255]))
        other_contour_color = tuple(int(v) for v in canny_cfg.get("other_contour_color_bgr", [128, 128, 128]))
        best_circle_color = tuple(int(v) for v in canny_cfg.get("best_circle_color_bgr", [0, 255, 0]))
        other_circle_color = tuple(int(v) for v in canny_cfg.get("other_circle_color_bgr", [0, 0, 255]))
        top5_circle_color = tuple(int(v) for v in canny_cfg.get("top5_circle_color_bgr", [255, 0, 255]))
        others_circle_color = tuple(int(v) for v in canny_cfg.get("others_circle_color_bgr", [128, 128, 128]))

        yolo_cfg = dict(config.get("yolo_debug", {}))
        yolo_map = None
        yolo_dets_file = yolo_cfg.get("yolo_detections_file")
        if triptych_enabled and yolo_dets_file:
            p = Path(str(yolo_dets_file))
            if p.exists():
                with open(p, "r") as f:
                    yolo_map = json.load(f)  # { "frame_idx": [ {Detection...}, ... ] }

        # Load geom detections saved by perception_geom step for overlay in debug panels.
        geom_map = None
        geom_dets_file = canny_cfg.get("geom_detections_file")
        if geom_dets_file:
            p = Path(str(geom_dets_file))
            if p.exists():
                with open(p, "r") as f:
                    geom_map = json.load(f)  # { "frame_idx": [ {Detection...}, ... ] }
        
        print(f"Rendering overlay video from {source_path} to {output_file}...")
        if canny_enabled:
            print(f"Rendering side-by-side Canny debug video to {side_by_side_file}...")
        if triptych_enabled:
            print(f"Rendering triptych debug video to {triptych_file}...")
        
        cap = cv2.VideoCapture(str(source_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        out_sbs = None
        if canny_enabled and side_by_side_file is not None:
            out_sbs = cv2.VideoWriter(str(side_by_side_file), fourcc, fps, (width * 2, height))
            if not out_sbs.isOpened():
                raise RuntimeError(f"Failed to open side-by-side writer: {side_by_side_file}")

        out_trip = None
        if triptych_enabled and triptych_file is not None:
            out_trip = cv2.VideoWriter(str(triptych_file), fourcc, fps, (width * 3, height))
            if not out_trip.isOpened():
                raise RuntimeError(f"Failed to open triptych writer: {triptych_file}")
        
        frame_idx = 0
        # Color palette (B, G, R) for classes or Track IDs
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

        base_label_scale = 0.5
        base_title_scale = 1.0
        base_thickness = 2
        label_scale = float(base_label_scale) * float(font_scale_mult)
        title_scale = float(base_title_scale) * float(font_scale_mult)
        thickness = int(max(1, round(float(base_thickness) * float(font_scale_mult))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = frame.copy()
            
            items = input_data.get(frame_idx, [])
            
            if items:
                overlay = frame.copy()
                
                for item in items:
                    # Determine if it's a Detection or TrackedObject
                    det = None
                    track_id = None
                    history = []
                    
                    if isinstance(item, TrackedObject):
                        det = item.detection
                        track_id = item.track_id
                        history = item.history
                    else:
                        # Assume Detection
                        det = item
                        
                    # Color Logic
                    if track_id is not None:
                        color = colors[track_id % len(colors)]
                    else:
                        color = colors[det.class_id % len(colors)]
                    
                    # 1. Draw Segmentation Mask
                    if det.mask:
                        pts = np.array(det.mask, dtype=np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], color)
                    
                    # 2. Draw Skeleton (Pose)
                    if det.keypoints:
                        kpts = det.keypoints
                        for kpt in kpts:
                            x, y, conf = kpt
                            if conf > 0.5:
                                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                        
                        for i, j in self.COCO_SKELETON:
                            if i < len(kpts) and j < len(kpts):
                                pt1 = kpts[i]
                                pt2 = kpts[j]
                                if pt1[2] > 0.5 and pt2[2] > 0.5:
                                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                                             (int(pt2[0]), int(pt2[1])), color, 2)

                    # If this is the athlete but we don't have keypoints (filled/predicted bbox),
                    # skip drawing to avoid a frozen/stuck skeleton/box on screen.
                    if det.class_name in ("atleta", "person") and not det.keypoints:
                        continue

                    # 3. Draw Bounding Box
                    x1, y1, x2, y2 = map(int, det.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # 4. Draw Label & Track ID
                    label = f"{det.class_name}"
                    if track_id is not None:
                        label += f" ID:{track_id}"
                    
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_scale,
                        color,
                        thickness,
                    )
                                
                    # 5. Draw Trajectory (History)
                    if history:
                        for i in range(1, len(history)):
                            pt1 = tuple(map(int, history[i-1]))
                            pt2 = tuple(map(int, history[i]))
                            cv2.line(frame, pt1, pt2, color, thickness)
                
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            out.write(frame)

            # Optional side-by-side debug output: left is the rendered overlay frame; right is Canny edges (+ geom dets).
            if out_sbs is not None:
                left = frame.copy()
                gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                if blur_ksize >= 3 and blur_ksize % 2 == 1:
                    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
                if clahe is not None:
                    gray = clahe.apply(gray)
                edges = cv2.Canny(gray, threshold1=canny1, threshold2=canny2)
                if morph_close:
                    k = max(1, int(morph_close_ksize))
                    if k % 2 == 0:
                        k += 1
                    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
                if invert_edges:
                    edges = cv2.bitwise_not(edges)
                right = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                # Draw geom detections on the canny panel if available
                if geom_map is not None:
                    dets = geom_map.get(str(frame_idx), [])
                    for d in dets:
                        try:
                            bbox = d.get("bbox", None)
                            if bbox is None:
                                continue
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(right, (x1, y1), (x2, y2), geom_color, 2)
                            cv2.putText(
                                right,
                                f"geom:{d.get('confidence', 0):.2f}",
                                (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                geom_color,
                                2,
                            )
                        except Exception:
                            continue

                # Titles
                cv2.putText(left, "overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), thickness)
                cv2.putText(right, f"canny ({canny1},{canny2})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), thickness)
                combined = np.concatenate([left, right], axis=1)
                out_sbs.write(combined)

            # Optional triptych debug output: overlay | canny+contours+circles | yolo raw
            if out_trip is not None:
                p1 = frame.copy()
                # Frame index (helps future debugging)
                cv2.putText(
                    p1,
                    f"frame={frame_idx}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.55, 0.85 * font_scale_mult),
                    (255, 255, 255),
                    thickness,
                )

                # Panel 2: Canny edges + geom circle detections (top-1 only)
                edges2 = self._compute_canny_edges(raw_frame, canny_cfg)
                if bool(canny_cfg.get("invert_edges", False)):
                    edges2 = cv2.bitwise_not(edges2)
                p2 = cv2.cvtColor(edges2, cv2.COLOR_GRAY2BGR)

                # Draw the best geom circle candidate (top-1)
                drew_any = False
                if geom_map is not None:
                    dets = geom_map.get(str(frame_idx), [])
                    dets = dets[:1]  # Only top-1
                    for d in dets:
                        try:
                            bbox = d.get("bbox", None)
                            if bbox is None:
                                continue
                            x1, y1, x2, y2 = bbox
                            cx = 0.5 * (float(x1) + float(x2))
                            cy = 0.5 * (float(y1) + float(y2))
                            r = d.get("radius_px", None)
                            if r is None:
                                r = 0.25 * (abs(float(x2) - float(x1)) + abs(float(y2) - float(y1)))
                            r = float(r)

                            cv2.circle(p2, (int(round(cx)), int(round(cy))), int(round(r)), best_circle_color, thickness)

                            # Diagnostics for best candidate
                            dbg = d.get("debug", None)
                            if isinstance(dbg, dict):
                                try:
                                    sc = float(d.get("confidence", d.get("shape_score", 0.0)))
                                    cf = float(dbg.get("contrast_frac", 0.0))
                                    ied = float(dbg.get("interior_edge_density", 0.0))
                                    arc = float(dbg.get("arc_frac", 0.0))
                                    txt = f"s={sc:.2f} arc={arc:.2f} c={cf:.2f} ied={ied:.3f}"
                                    cv2.putText(
                                        p2,
                                        txt,
                                        (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        max(0.55, 0.85 * font_scale_mult),
                                        (255, 255, 255),
                                        thickness,
                                    )
                                except Exception:
                                    pass
                            drew_any = True
                        except Exception:
                            continue

                # Fallback: contour-based visualization if no geom detections
                if not drew_any:
                    edges_fallback, candidates = self._compute_canny_contours_and_circles(raw_frame, canny_cfg)
                    if bool(canny_cfg.get("invert_edges", False)):
                        edges_fallback = cv2.bitwise_not(edges_fallback)
                    p2 = cv2.cvtColor(edges_fallback, cv2.COLOR_GRAY2BGR)
                    if candidates:
                        c = candidates[0]  # Best only
                        cnt = c["contour"]
                        x, y, r = c["circle"]
                        cv2.drawContours(p2, [cnt], -1, best_contour_color, thickness)
                        cv2.circle(p2, (int(round(x)), int(round(y))), int(round(r)), best_circle_color, thickness)

                # Panel 3: yolo raw on original frame
                p3 = raw_frame.copy()
                if yolo_map is not None:
                    dets = yolo_map.get(str(frame_idx), [])
                    for i, d in enumerate(dets):
                        try:
                            bbox = d.get("bbox", None)
                            if bbox is None:
                                continue
                            x1, y1, x2, y2 = map(int, bbox)
                            cls = str(d.get("class_name", "unknown"))
                            conf = float(d.get("confidence", 0.0))
                            color = colors[int(d.get("class_id", 0)) % len(colors)]
                            cv2.rectangle(p3, (x1, y1), (x2, y2), color, thickness)
                            label = f"{cls} idx:{i} conf:{conf:.2f}"
                            cv2.putText(
                                p3,
                                label,
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                label_scale,
                                color,
                                thickness,
                            )
                        except Exception:
                            continue

                # Titles
                cv2.putText(p1, "overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), thickness)
                cv2.putText(p2, "canny + geom(top1)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), thickness)
                cv2.putText(p3, "yolo raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), thickness)

                combined3 = np.concatenate([p1, p2, p3], axis=1)
                out_trip.write(combined3)

            frame_idx += 1
            if frame_idx % 20 == 0:
                print(f"Rendered frame {frame_idx}...", end='\r')
                
        cap.release()
        out.release()
        if out_sbs is not None:
            out_sbs.release()
        if out_trip is not None:
            out_trip.release()
        print(f"\nOverlay video saved to {output_file}")
        if side_by_side_file is not None and canny_enabled:
            print(f"Side-by-side Canny debug video saved to {side_by_side_file}")
        if triptych_file is not None and triptych_enabled:
            print(f"Triptych debug video saved to {triptych_file}")
        return output_file

    def save_result(self, data: Path, output_path: Path) -> None:
        with open(output_path, 'w') as f:
            f.write(str(data))

    def load_result(self, input_path: Path) -> Path:
        with open(input_path, 'r') as f:
            return Path(f.read().strip())
