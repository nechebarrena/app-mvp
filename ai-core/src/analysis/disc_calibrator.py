import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from domain.ports import IPipelineStep
from domain.entities import TrackedObject


class DiscConsensusCalibrator(IPipelineStep[Dict[int, List[TrackedObject]], Dict[str, Any]]):
    """
    Builds a robust initial disc prior (center/radius/uncertainty) from consensus between
    YOLO and geometric detections in the first N frames.

    Expects tracked objects containing detections with:
      - class_name == disc_class_name (default: "discos")
      - detection.source (e.g., "yolo", "geom")

    Output dict is small and serializable; it is meant to be consumed by DiscFusionTracker.
    """

    @staticmethod
    def _center_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    @staticmethod
    def _radius_from_detection(obj: TrackedObject) -> float:
        det = obj.detection
        if det.radius_px is not None:
            return float(det.radius_px)
        x1, y1, x2, y2 = det.bbox
        w = max(1e-6, float(x2 - x1))
        h = max(1e-6, float(y2 - y1))
        # equivalent radius from bbox (rough)
        return float(0.25 * (w + h))

    @staticmethod
    def _mad_sigma(values: List[float]) -> float:
        if not values:
            return 0.0
        med = float(np.median(values))
        mad = float(np.median([abs(v - med) for v in values]))
        # normal approx
        return float(1.4826 * mad)

    @staticmethod
    def _select_primary_athlete(frames: Dict[int, List[TrackedObject]], athlete_class_name: str) -> Dict[int, TrackedObject]:
        per_tid: Dict[int, List[Tuple[int, TrackedObject]]] = {}
        for f, objs in frames.items():
            for o in objs:
                if o.detection.class_name == athlete_class_name:
                    per_tid.setdefault(o.track_id, []).append((f, o))
        if not per_tid:
            return {}
        best_tid = max(per_tid.keys(), key=lambda tid: len(per_tid[tid]))
        return {f: o for (f, o) in per_tid[best_tid]}

    @staticmethod
    def _wrists_mid(ath: Optional[TrackedObject], min_conf: float, require_both: bool) -> Optional[Tuple[float, float]]:
        if ath is None or not ath.detection.keypoints:
            return None
        k = ath.detection.keypoints
        pts: List[Tuple[float, float]] = []
        for idx in (9, 10):  # COCO wrists
            if idx < len(k):
                x, y, c = k[idx]
                if float(c) >= float(min_conf):
                    pts.append((float(x), float(y)))
        if require_both and len(pts) < 2:
            return None
        if not pts:
            return None
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        return (mx, my)

    @staticmethod
    def _group_by_source(objs: List[TrackedObject], disc_class_name: str) -> Dict[str, List[TrackedObject]]:
        """Group detections by their source (yolo, geom, etc.)."""
        groups: Dict[str, List[TrackedObject]] = {}
        for o in objs:
            if o.detection.class_name == disc_class_name:
                src = o.detection.source or "unknown"
                groups.setdefault(src, []).append(o)
        return groups

    def _score_disc_candidate(
        self,
        frame_idx: int,
        disc: TrackedObject,
        athlete_ref: Dict[int, TrackedObject],
        wrist_penalty_weight: float,
        wrist_penalty_scale_px: float,
        wrist_min_conf: float,
        wrist_require_both: bool,
    ) -> Tuple[float, Optional[float]]:
        """
        Returns (score, wrist_dist_px_or_None).
        Score is confidence minus a wrist-distance penalty (near-hard gate when weight is large).
        If wrists are not available for the frame, no penalty is applied.
        """
        base = float(disc.detection.confidence)
        ath = athlete_ref.get(frame_idx)
        wm = self._wrists_mid(ath, wrist_min_conf, wrist_require_both)
        if wm is None:
            return base, None
        cx, cy = self._center_from_bbox(disc.detection.bbox)
        d = float(np.hypot(cx - wm[0], cy - wm[1]))
        penalty = float(wrist_penalty_weight) * (d / max(1.0, float(wrist_penalty_scale_px)))
        return base - penalty, d

    def _find_best_in_source(
        self,
        candidates: List[TrackedObject],
        frame_idx: int,
        athlete_ref: Dict[int, TrackedObject],
        min_seed_radius_px: float,
        wrist_penalty_weight: float,
        wrist_penalty_scale_px: float,
        wrist_min_conf: float,
        wrist_require_both: bool,
    ) -> Optional[Tuple[float, TrackedObject]]:
        """Find the best candidate within a single source, filtering by radius and scoring."""
        scored = []
        for o in candidates:
            if self._radius_from_detection(o) < min_seed_radius_px:
                continue
            score, _ = self._score_disc_candidate(
                frame_idx, o, athlete_ref,
                wrist_penalty_weight, wrist_penalty_scale_px, wrist_min_conf, wrist_require_both
            )
            scored.append((score, o))
        if not scored:
            return None
        return max(scored, key=lambda t: t[0])

    def _try_pairwise_consensus(
        self,
        src_a_objs: List[TrackedObject],
        src_b_objs: List[TrackedObject],
        frame_idx: int,
        athlete_ref: Dict[int, TrackedObject],
        min_seed_radius_px: float,
        max_center_dist: float,
        max_radius_rel_diff: float,
        wrist_penalty_weight: float,
        wrist_penalty_scale_px: float,
        wrist_min_conf: float,
        wrist_require_both: bool,
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        Try to find consensus between two source lists for a single frame.
        Returns (center, radius) if consensus found, else None.
        """
        best_a = self._find_best_in_source(
            src_a_objs, frame_idx, athlete_ref, min_seed_radius_px,
            wrist_penalty_weight, wrist_penalty_scale_px, wrist_min_conf, wrist_require_both
        )
        best_b = self._find_best_in_source(
            src_b_objs, frame_idx, athlete_ref, min_seed_radius_px,
            wrist_penalty_weight, wrist_penalty_scale_px, wrist_min_conf, wrist_require_both
        )
        if best_a is None or best_b is None:
            return None

        _, obj_a = best_a
        _, obj_b = best_b

        ca = self._center_from_bbox(obj_a.detection.bbox)
        cb = self._center_from_bbox(obj_b.detection.bbox)
        dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
        if dist > max_center_dist:
            return None

        ra = self._radius_from_detection(obj_a)
        rb = self._radius_from_detection(obj_b)
        rel_diff = float(abs(ra - rb) / max(1e-6, max(ra, rb)))
        if rel_diff > max_radius_rel_diff:
            return None

        # Consensus: midpoint
        center = ((ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0)
        radius = (ra + rb) / 2.0
        return (center, radius)

    def run(self, input_data: Dict[int, List[TrackedObject]], config: Dict[str, Any]) -> Dict[str, Any]:
        calibration_frames = int(config.get("calibration_frames", 30))
        disc_class_name = str(config.get("disc_class_name", "discos"))
        max_center_dist = float(config.get("match_max_center_dist_px", 80.0))
        max_radius_rel_diff = float(config.get("match_max_radius_rel_diff", 0.25))
        # Prevent seeding on small inner rings / background circles
        min_seed_radius_px = float(config.get("min_seed_radius_px", 80.0))

        # Optional: priority order for fallback (if no consensus)
        fallback_source_priority: Optional[List[str]] = config.get("fallback_source_priority", None)

        # Wrist-distance penalty (soft, but can be made near-hard by using a large weight)
        athlete_class_name = str(config.get("athlete_class_name", "person"))
        wrist_penalty_weight = float(config.get("wrist_penalty_weight", 12.0))
        wrist_penalty_scale_px = float(config.get("wrist_penalty_scale_px", 250.0))
        wrist_min_conf = float(config.get("wrist_min_conf", 0.25))
        wrist_require_both = bool(config.get("wrist_require_both", False))

        frames_sorted = sorted(input_data.keys())
        frames_window = frames_sorted[:calibration_frames]

        matched_centers: List[Tuple[float, float]] = []
        matched_radii: List[float] = []
        matched_frames: List[int] = []
        consensus_pair: Optional[Tuple[str, str]] = None

        athlete_ref = self._select_primary_athlete(input_data, athlete_class_name)

        # Discover available sources across all calibration frames
        all_sources: Dict[str, int] = {}  # source -> detection count
        for f in frames_window:
            objs = input_data.get(f, [])
            groups = self._group_by_source(objs, disc_class_name)
            for src, dets in groups.items():
                all_sources[src] = all_sources.get(src, 0) + len(dets)

        source_names = sorted(all_sources.keys(), key=lambda s: -all_sources[s])  # most detections first

        # Multi-source mode: try pairwise consensus if 2+ sources
        if len(source_names) >= 2:
            for src_a, src_b in combinations(source_names, 2):
                for f in frames_window:
                    objs = input_data.get(f, [])
                    groups = self._group_by_source(objs, disc_class_name)
                    if src_a not in groups or src_b not in groups:
                        continue

                    result = self._try_pairwise_consensus(
                        groups[src_a], groups[src_b], f, athlete_ref,
                        min_seed_radius_px, max_center_dist, max_radius_rel_diff,
                        wrist_penalty_weight, wrist_penalty_scale_px, wrist_min_conf, wrist_require_both
                    )
                    if result is not None:
                        center, radius = result
                        matched_centers.append(center)
                        matched_radii.append(radius)
                        matched_frames.append(int(f))
                        if consensus_pair is None:
                            consensus_pair = (src_a, src_b)

                # If we found matches with this pair, stop trying other pairs
                if matched_centers:
                    break

        result: Dict[str, Any]

        if not matched_centers:
            # Fallback: use best candidate from any available source
            # Prioritize by fallback_source_priority if provided, else by detection count
            if fallback_source_priority:
                priority_order = [s for s in fallback_source_priority if s in source_names]
                priority_order.extend([s for s in source_names if s not in priority_order])
            else:
                priority_order = source_names

            best = None
            best_score = -float("inf")
            best_wrist_dist = None
            best_source = None

            for src in priority_order:
                for f in frames_window:
                    objs = input_data.get(f, [])
                    groups = self._group_by_source(objs, disc_class_name)
                    if src not in groups:
                        continue
                    for o in groups[src]:
                        if self._radius_from_detection(o) < min_seed_radius_px:
                            continue
                        score, wd = self._score_disc_candidate(
                            f, o, athlete_ref, wrist_penalty_weight, wrist_penalty_scale_px, wrist_min_conf, wrist_require_both
                        )
                        if best is None or score > best_score:
                            best = o
                            best_score = score
                            best_wrist_dist = wd
                            best_source = src

            if best is None:
                result = {
                    "ok": False,
                    "reason": "no_consensus_and_no_fallback_seed",
                    "disc_class_name": disc_class_name,
                    "calibration_frames": calibration_frames,
                    "sources_found": source_names,
                }
                self._maybe_write_prior_artifact(result, config)
                return result

            c = self._center_from_bbox(best.detection.bbox)
            r = self._radius_from_detection(best)
            result = {
                "ok": True,
                "mode": f"fallback_single_source",
                "fallback_source": best_source,
                "disc_class_name": disc_class_name,
                "calibration_frames": calibration_frames,
                "seed_center": [c[0], c[1]],
                "seed_radius": float(r),
                "sigma_center": float(50.0),
                "sigma_radius": float(0.25 * r),
                "matched_frames": [],
                "matched_count": 0,
                "min_seed_radius_px": min_seed_radius_px,
                "sources_found": source_names,
                "wrist_penalty": {
                    "enabled": True,
                    "athlete_class_name": athlete_class_name,
                    "wrist_penalty_weight": wrist_penalty_weight,
                    "wrist_penalty_scale_px": wrist_penalty_scale_px,
                    "wrist_min_conf": wrist_min_conf,
                    "wrist_require_both": wrist_require_both,
                    "seed_wrist_dist_px": best_wrist_dist,
                },
            }
            self._maybe_write_prior_artifact(result, config)
            return result

        xs = [c[0] for c in matched_centers]
        ys = [c[1] for c in matched_centers]
        rs = matched_radii

        seed_center = [float(np.median(xs)), float(np.median(ys))]
        seed_radius = float(np.median(rs))
        sigma_x = self._mad_sigma(xs)
        sigma_y = self._mad_sigma(ys)
        sigma_center = float(max(sigma_x, sigma_y))
        sigma_radius = self._mad_sigma(rs)

        result = {
            "ok": True,
            "mode": "consensus",
            "consensus_sources": list(consensus_pair) if consensus_pair else source_names[:2],
            "disc_class_name": disc_class_name,
            "calibration_frames": calibration_frames,
            "seed_center": seed_center,
            "seed_radius": seed_radius,
            "sigma_center": sigma_center,
            "sigma_radius": sigma_radius,
            "matched_frames": matched_frames,
            "matched_count": int(len(matched_frames)),
            "min_seed_radius_px": min_seed_radius_px,
            "sources_found": source_names,
            "params": {
                "match_max_center_dist_px": max_center_dist,
                "match_max_radius_rel_diff": max_radius_rel_diff,
            },
            "wrist_penalty": {
                "enabled": True,
                "athlete_class_name": athlete_class_name,
                "wrist_penalty_weight": wrist_penalty_weight,
                "wrist_penalty_scale_px": wrist_penalty_scale_px,
                "wrist_min_conf": wrist_min_conf,
                "wrist_require_both": wrist_require_both,
            },
        }
        self._maybe_write_prior_artifact(result, config)
        return result

    def _maybe_write_prior_artifact(self, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Writes a stable-named artifact `disc_prior.json` into the run output directory,
        in addition to the standard PipelineRunner save_output path.
        """
        out_dir = config.get("_run_output_dir")
        if not out_dir:
            return
        try:
            p = Path(out_dir).resolve() / "disc_prior.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Do not fail calibration due to artifact writing
            pass

    def save_result(self, data: Dict[str, Any], output_path: Path) -> None:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_result(self, input_path: Path) -> Dict[str, Any]:
        with open(input_path, "r") as f:
            return json.load(f)


