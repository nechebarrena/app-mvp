import json
import logging
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from domain.ports import IPipelineStep
from domain.entities import TrackedObject, Detection
from analysis.tracking_utils import KalmanFilter2D


@dataclass
class _Candidate:
    source: str
    det: Detection
    frame: int
    center: Tuple[float, float]
    radius: float
    conf: float
    shape: float


class DiscFusionTracker(IPipelineStep[List[Any], Dict[int, List[TrackedObject]]]):
    """
    Tracks the disc using multi-source measurements (YOLO, geom) and Kalman prediction.

    Input is expected to be a list of:
      - tracking_map: Dict[int, List[TrackedObject]]
      - disc_prior: Dict[str, Any] from DiscConsensusCalibrator
    (order as configured in YAML input_from_step list)

    Output: Dict[frame_idx, List[TrackedObject]] containing
      - the selected athlete (if available) and
      - the fused disc track with class_name="discos"
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("DiscFusionTracker")

    @staticmethod
    def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    @staticmethod
    def _radius_from_det(det: Detection) -> float:
        if det.radius_px is not None:
            return float(det.radius_px)
        x1, y1, x2, y2 = det.bbox
        w = max(1e-6, float(x2 - x1))
        h = max(1e-6, float(y2 - y1))
        return float(0.25 * (w + h))

    def _extract_candidates(
        self,
        tracking_map: Dict[int, List[TrackedObject]],
        frame: int,
        disc_class_name: str,
    ) -> List[_Candidate]:
        out: List[_Candidate] = []
        for obj in tracking_map.get(frame, []):
            det = obj.detection
            if det.class_name != disc_class_name:
                continue
            src = det.source or "unknown"
            c = self._center(det.bbox)
            r = self._radius_from_det(det)
            out.append(
                _Candidate(
                    source=src,
                    det=det,
                    frame=frame,
                    center=c,
                    radius=r,
                    conf=float(det.confidence),
                    shape=float(det.shape_score) if det.shape_score is not None else 0.0,
                )
            )
        return out

    def _select_primary_athlete(self, tracking_map: Dict[int, List[TrackedObject]], athlete_class_name: str) -> Dict[int, TrackedObject]:
        # pick the most persistent track_id for athlete class
        per_tid: Dict[int, List[Tuple[int, TrackedObject]]] = {}
        for f, objs in tracking_map.items():
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

    def _discover_sources(
        self,
        tracking_map: Dict[int, List[TrackedObject]],
        disc_class_name: str,
    ) -> Dict[str, int]:
        """Discover available detection sources and count detections per source."""
        counts: Dict[str, int] = {}
        for objs in tracking_map.values():
            for o in objs:
                if o.detection.class_name == disc_class_name:
                    src = o.detection.source or "unknown"
                    counts[src] = counts.get(src, 0) + 1
        return counts

    def _select_consensus_sources(
        self,
        config: Dict[str, Any],
        available_sources: Dict[str, int],
    ) -> List[str]:
        """
        Select sources for consensus override.
        If consensus_sources is a list, use it (filtering to available ones).
        If "auto", pick top 2 sources by detection count.
        """
        cfg_val = config.get("consensus_sources", ["yolo", "geom"])
        if cfg_val == "auto":
            sorted_sources = sorted(available_sources.keys(), key=lambda s: -available_sources[s])
            return sorted_sources[:2]
        elif isinstance(cfg_val, list):
            return [s for s in cfg_val if s in available_sources]
        else:
            # Backwards compat: if not specified, use old defaults
            return [s for s in ["yolo", "geom"] if s in available_sources]

    def run(self, input_data: List[Any], config: Dict[str, Any]) -> Dict[int, List[TrackedObject]]:
        if not isinstance(input_data, list) or len(input_data) != 2:
            raise ValueError("DiscFusionTracker expects input_data=[tracking_map, disc_prior]")

        tracking_map: Dict[int, List[TrackedObject]] = input_data[0]
        disc_prior: Dict[str, Any] = input_data[1]

        run_output_dir = Path(config.get("_run_output_dir", ".")).resolve()
        debug_enabled = bool(config.get("debug", True))
        debug_jsonl = run_output_dir / "disc_tracker_debug.jsonl"
        debug_summary = run_output_dir / "disc_tracker_debug_summary.json"

        # Params
        disc_class_name = str(config.get("disc_class_name", disc_prior.get("disc_class_name", "discos")))
        athlete_class_name = str(config.get("athlete_class_name", "person"))

        use_athlete_roi = bool(config.get("use_athlete_roi", True))
        roi_expand_factor = float(config.get("roi_expand_factor", 1.8))

        # Wrist penalty (soft, but can be made near-hard by using a large weight)
        use_wrist_penalty = bool(config.get("use_wrist_penalty", True))
        wrist_penalty_weight = float(config.get("wrist_penalty_weight", 20.0))
        wrist_penalty_scale_px = float(config.get("wrist_penalty_scale_px", 250.0))
        wrist_min_conf = float(config.get("wrist_min_conf", 0.25))
        wrist_require_both = bool(config.get("wrist_require_both", False))

        # gating
        base_radius_px = float(config.get("search_radius_px", 120.0))
        radius_sigma_mult_tracking = float(config.get("radius_sigma_mult_tracking", 2.0))
        radius_sigma_mult_lost = float(config.get("radius_sigma_mult_lost", 3.5))
        dist_sigma_mult_tracking = float(config.get("dist_sigma_mult_tracking", 3.0))
        dist_sigma_mult_lost = float(config.get("dist_sigma_mult_lost", 5.0))
        # Optional: gate distance using Kalman covariance instead of heuristic sigma_center
        use_kalman_cov_gate = bool(config.get("use_kalman_cov_gate", True))
        cov_gate_mult_tracking = float(config.get("cov_gate_mult_tracking", 3.0))
        cov_gate_mult_lost = float(config.get("cov_gate_mult_lost", 6.0))

        lost_enter_frames = int(config.get("lost_enter_frames", 3))
        lost_stop_frames = int(config.get("lost_stop_frames", 60))
        stop_on_lost = bool(config.get("stop_on_lost", False))

        # weights
        w_dist = float(config.get("w_dist", 1.0))
        w_rad = float(config.get("w_radius", 2.0))
        w_shape = float(config.get("w_shape", 1.0))

        # Source biases: dict mapping source -> cost bias (negative = prefer, positive = penalize)
        # Backwards compat: if source_biases not present, use old w_source_yolo_bias
        source_biases_cfg = config.get("source_biases", None)
        if source_biases_cfg is not None and isinstance(source_biases_cfg, dict):
            source_biases: Dict[str, float] = {str(k): float(v) for k, v in source_biases_cfg.items()}
        else:
            # Backwards compat: only YOLO has a bias
            w_source_yolo_bias = float(config.get("w_source_yolo_bias", -0.15))
            source_biases = {"yolo": w_source_yolo_bias}

        # Acceptance thresholds: if the best candidate is still too costly, stay LOST rather than latching to noise.
        max_accept_cost_tracking = float(config.get("max_accept_cost_tracking", 10.0))
        max_accept_cost_lost = float(config.get("max_accept_cost_lost", 15.0))
        emit_prediction_when_lost = bool(config.get("emit_prediction_when_lost", True))
        disc_class_id = int(config.get("disc_class_id", 2))

        # --- MVP selection strategy (top-1 + anti-static + anti-jump) ---
        # top1_sources: list of sources for which only top-1 candidate is kept
        # Backwards compat: if not present, use use_geom_top1_only for ["geom_ransac"]
        top1_sources_cfg = config.get("top1_sources", None)
        if top1_sources_cfg is not None and isinstance(top1_sources_cfg, list):
            top1_sources: List[str] = [str(s) for s in top1_sources_cfg]
        else:
            use_geom_top1_only = bool(config.get("use_geom_top1_only", False))
            top1_sources = ["geom"] if use_geom_top1_only else []

        static_window = int(config.get("static_window", 25))
        static_center_std_px = float(config.get("static_center_std_px", 3.0))
        static_radius_std_px = float(config.get("static_radius_std_px", 2.0))
        jump_thresh_px = float(config.get("jump_thresh_px", 80.0))

        # Consensus override: configurable sources (list or "auto")
        use_consensus_override = bool(config.get("use_consensus_override", True))
        consensus_eps_px = float(config.get("consensus_eps_px", 30.0))

        # Kalman noise
        process_noise = float(config.get("kalman_process_noise", 1.0))
        measurement_noise = float(config.get("kalman_measurement_noise", 10.0))

        # prior
        if not disc_prior.get("ok"):
            self._logger.warning(f"DiscFusionTracker: prior not ok: {disc_prior}")

        seed_center = disc_prior.get("seed_center", None)
        seed_radius = float(disc_prior.get("seed_radius", 0.0))
        prior_radius = float(seed_radius)
        sigma_center = float(disc_prior.get("sigma_center", 50.0))
        sigma_radius = float(disc_prior.get("sigma_radius", max(10.0, 0.25 * max(1.0, seed_radius))))

        if seed_center is None:
            # fallback: use first available candidate
            for f in sorted(tracking_map.keys()):
                cands = self._extract_candidates(tracking_map, f, disc_class_name)
                if cands:
                    seed_center = [cands[0].center[0], cands[0].center[1]]
                    seed_radius = cands[0].radius
                    break

        if seed_center is None:
            return {}

        # athlete reference (optional)
        athlete_ref = self._select_primary_athlete(tracking_map, athlete_class_name)

        # Discover available sources and select consensus sources
        available_sources = self._discover_sources(tracking_map, disc_class_name)
        consensus_sources = self._select_consensus_sources(config, available_sources) if use_consensus_override else []

        def roi_for_frame(f: int) -> Optional[Tuple[float, float, float, float]]:
            if not use_athlete_roi:
                return None
            a = athlete_ref.get(f)
            if a is None:
                return None
            x1, y1, x2, y2 = a.detection.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = (x2 - x1)
            h = (y2 - y1)
            mx = roi_expand_factor * w
            my = roi_expand_factor * h
            return (cx - mx, cy - my, cx + mx, cy + my)

        # debug init
        if debug_enabled:
            debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_jsonl, "w", encoding="utf-8") as f:
                f.write(json.dumps({"event": "run_start", "mode": "fusion"}, ensure_ascii=False) + "\n")

        kf = KalmanFilter2D(
            x=float(seed_center[0]),
            y=float(seed_center[1]),
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )

        master_id = int(config.get("master_track_id", 777))
        frames_sorted = sorted(tracking_map.keys())
        state = "TRACKING"
        lost_counter = 0

        outputs: Dict[int, List[TrackedObject]] = {}
        disc_track: List[TrackedObject] = []

        # summary counters
        reason_counts = {"roi": 0, "dist": 0, "radius": 0, "empty": 0, "cost": 0, "static_like": 0, "jump": 0}
        used_sources = {"yolo": 0, "geom": 0, "fused": 0, "pred": 0, "unknown": 0}
        first_lost_frame: Optional[int] = None

        # static/jump helper state
        recent_meas = deque(maxlen=max(1, static_window))
        last_meas_center: Optional[Tuple[float, float]] = None

        for f in frames_sorted:
            pred_x, pred_y = kf.predict()
            roi = roi_for_frame(f)
            wm = self._wrists_mid(athlete_ref.get(f), wrist_min_conf, wrist_require_both) if use_wrist_penalty else None

            cands = self._extract_candidates(tracking_map, f, disc_class_name)
            topk = []
            best: Optional[_Candidate] = None
            best_cost = float("inf")
            rejected_reason: Optional[str] = None
            geom_top1: Optional[Dict[str, Any]] = None

            valid_scored: List[Tuple[_Candidate, float]] = []

            dist_sigma_mult = dist_sigma_mult_tracking if state == "TRACKING" else dist_sigma_mult_lost
            rad_sigma_mult = radius_sigma_mult_tracking if state == "TRACKING" else radius_sigma_mult_lost

            # Distance gate: either from Kalman covariance (preferred) or from heuristic sigma_center
            if use_kalman_cov_gate and hasattr(kf, "P"):
                try:
                    P = kf.P  # type: ignore[attr-defined]
                    sig_x = float(np.sqrt(max(1e-6, float(P[0, 0]))))
                    sig_y = float(np.sqrt(max(1e-6, float(P[1, 1]))))
                    sig_pos = float(np.hypot(sig_x, sig_y))
                    cov_mult = cov_gate_mult_tracking if state == "TRACKING" else cov_gate_mult_lost
                    dist_gate = max(base_radius_px, float(cov_mult) * sig_pos)
                except Exception:
                    dist_gate = max(base_radius_px, dist_sigma_mult * sigma_center)
            else:
                dist_gate = max(base_radius_px, dist_sigma_mult * sigma_center)
            rad_gate = max(10.0, rad_sigma_mult * sigma_radius)

            if not cands:
                reason_counts["empty"] += 1
            else:
                for cand in cands:
                    # ROI gate
                    if roi is not None:
                        if not (roi[0] <= cand.center[0] <= roi[2] and roi[1] <= cand.center[1] <= roi[3]):
                            topk.append({"source": cand.source, "reject": "roi", "center": list(cand.center), "radius": cand.radius})
                            continue

                    # distance gate
                    dist = float(np.hypot(cand.center[0] - pred_x, cand.center[1] - pred_y))
                    if dist > dist_gate:
                        reason_counts["dist"] += 1
                        topk.append({"source": cand.source, "reject": "dist", "dist": dist, "center": list(cand.center), "radius": cand.radius})
                        continue

                    # radius consistency gate (with seed_radius)
                    if seed_radius > 1e-6:
                        if abs(cand.radius - seed_radius) > rad_gate:
                            reason_counts["radius"] += 1
                            topk.append({"source": cand.source, "reject": "radius", "dist": dist, "radius": cand.radius, "seed_radius": seed_radius})
                            continue

                    # cost (lower is better)
                    rad_res = 0.0 if seed_radius <= 1e-6 else abs(cand.radius - seed_radius) / max(1e-6, seed_radius)
                    shape_pen = 0.0
                    if cand.source == "geom":
                        # prefer higher circularity -> lower penalty
                        shape_pen = float(max(0.0, 1.0 - cand.shape))

                    # Source-specific bias (configurable per source)
                    src_bias = source_biases.get(cand.source, 0.0)

                    wrist_dist = None
                    wrist_pen = 0.0
                    if wm is not None:
                        wrist_dist = float(np.hypot(cand.center[0] - wm[0], cand.center[1] - wm[1]))
                        wrist_pen = float(wrist_penalty_weight) * (wrist_dist / max(1.0, float(wrist_penalty_scale_px)))

                    cost = (
                        (w_dist * (dist / max(1e-6, dist_gate)))
                        + (w_rad * rad_res)
                        + (w_shape * shape_pen)
                        + src_bias
                        + wrist_pen
                    )

                    topk.append(
                        {
                            "source": cand.source,
                            "reject": None,
                            "dist": dist,
                            "dist_gate": dist_gate,
                            "radius": cand.radius,
                            "rad_gate": rad_gate,
                            "rad_res": rad_res,
                            "shape": cand.shape,
                            "cost": cost,
                            "wrist_dist": wrist_dist,
                            "wrist_penalty": wrist_pen,
                            "center": [cand.center[0], cand.center[1]],
                            "bbox": list(cand.det.bbox),
                            "conf": cand.conf,
                        }
                    )
                    valid_scored.append((cand, float(cost)))

            # For sources in top1_sources, keep only the best candidate (by cost), discard others
            # Also track top-1 per source for consensus override
            source_top1: Dict[str, Dict[str, Any]] = {}
            if valid_scored:
                # Group by source and find top-1 for each
                by_source: Dict[str, List[Tuple[_Candidate, float]]] = {}
                for c, cost in valid_scored:
                    by_source.setdefault(c.source, []).append((c, cost))

                for src, items in by_source.items():
                    c0, cost0 = min(items, key=lambda x: x[1])
                    source_top1[src] = {
                        "source": c0.source,
                        "center": [c0.center[0], c0.center[1]],
                        "radius": float(c0.radius),
                        "conf": float(c0.conf),
                        "shape": float(c0.shape),
                        "bbox": list(c0.det.bbox),
                        "cost": float(cost0),
                        "_cand": c0,  # Keep reference for filtering
                    }

                # Filter: for sources in top1_sources, keep only their top-1
                if top1_sources:
                    filtered = []
                    for c, cost in valid_scored:
                        if c.source in top1_sources:
                            # Only keep if this is the top-1 for this source
                            if source_top1.get(c.source, {}).get("_cand") is c:
                                filtered.append((c, cost))
                        else:
                            filtered.append((c, cost))
                    valid_scored = filtered

                # Backwards compat: expose geom_top1 for debug logging
                geom_top1 = source_top1.get("geom")
                if geom_top1:
                    geom_top1 = {k: v for k, v in geom_top1.items() if k != "_cand"}

                # Choose best overall by cost after optional filtering
                if valid_scored:
                    best, best_cost = min(valid_scored, key=lambda x: x[1])

            accepted = best is not None
            if accepted:
                cost_threshold = max_accept_cost_tracking if state == "TRACKING" else max_accept_cost_lost
                if best_cost > cost_threshold:
                    accepted = False
                    rejected_reason = "cost"
                    reason_counts["cost"] += 1

            # Consensus override: if configured consensus sources agree spatially, trust that over prediction.
            consensus_used = False
            consensus_center: Optional[Tuple[float, float]] = None
            consensus_sources_used: Optional[List[str]] = None
            if use_consensus_override and len(consensus_sources) >= 2:
                # Check if we have top-1 candidates from at least 2 consensus sources
                consensus_tops = [(src, source_top1[src]) for src in consensus_sources if src in source_top1]
                if len(consensus_tops) >= 2:
                    # Check pairwise distance between first two
                    src_a, top_a = consensus_tops[0]
                    src_b, top_b = consensus_tops[1]
                    dx = float(top_a["center"][0]) - float(top_b["center"][0])
                    dy = float(top_a["center"][1]) - float(top_b["center"][1])
                    if float(np.hypot(dx, dy)) <= float(consensus_eps_px):
                        consensus_used = True
                        consensus_sources_used = [src_a, src_b]
                        consensus_center = (
                            0.5 * (float(top_a["center"][0]) + float(top_b["center"][0])),
                            0.5 * (float(top_a["center"][1]) + float(top_b["center"][1]))
                        )
                        accepted = True
                        rejected_reason = None

            # Anti-jump: reject teleports vs last accepted measurement (stricter in TRACKING)
            if accepted and (not consensus_used) and last_meas_center is not None and best is not None:
                jump = float(np.hypot(best.center[0] - last_meas_center[0], best.center[1] - last_meas_center[1]))
                jt = float(jump_thresh_px) if state == "TRACKING" else float(2.0 * jump_thresh_px)
                if jump > jt:
                    accepted = False
                    rejected_reason = "jump"
                    reason_counts["jump"] += 1

            # Anti-static: if we have been essentially static for a full window, reject further updates and force LOST
            if accepted and static_window >= 3:
                if len(recent_meas) >= static_window:
                    arr = np.array(recent_meas, dtype=np.float32)  # (N,3) -> x,y,r
                    mu = np.mean(arr[:, :2], axis=0)
                    d = np.sqrt(np.sum((arr[:, :2] - mu[None, :]) ** 2, axis=1))
                    center_std = float(np.std(d))
                    rad_std = float(np.std(arr[:, 2]))
                    if center_std <= static_center_std_px and rad_std <= static_radius_std_px:
                        accepted = False
                        rejected_reason = "static_like"
                        reason_counts["static_like"] += 1
                        # Help reacquire elsewhere: inflate uncertainty and force LOST so dist_gate expands.
                        sigma_center = float(max(sigma_center, 500.0))
                        try:
                            kf.P = kf.P * 10.0  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        state = "LOST"
                        lost_counter = max(lost_counter, lost_enter_frames)

            if debug_enabled:
                # keep only top K entries for readability
                topk_sorted = sorted(topk, key=lambda x: (x["reject"] is not None, x.get("cost", 9999.0)))[: int(config.get("debug_topk", 6))]
                with open(debug_jsonl, "a", encoding="utf-8") as fjsonl:
                    fjsonl.write(
                        json.dumps(
                            {
                                "event": "frame_eval",
                                "frame": int(f),
                                "state": state,
                                "pred": [float(pred_x), float(pred_y)],
                                "roi": list(roi) if roi is not None else None,
                                "dist_gate": float(dist_gate),
                                "rad_gate": float(rad_gate),
                                "seed_radius": float(seed_radius),
                                "sigma_center": float(sigma_center),
                                "sigma_radius": float(sigma_radius),
                                "candidates_total": int(len(cands)),
                                "accepted": accepted,
                                "rejected_reason": rejected_reason,
                                "geom_top1": geom_top1,
                                "consensus_used": consensus_used,
                                "consensus_sources": consensus_sources_used,
                                "consensus_center": list(consensus_center) if consensus_center is not None else None,
                                "best": None
                                if best is None
                                else {
                                    "source": best.source,
                                    "center": [best.center[0], best.center[1]],
                                    "radius": best.radius,
                                    "conf": best.conf,
                                    "shape": best.shape,
                                    "bbox": list(best.det.bbox),
                                    "cost": float(best_cost),
                                },
                                "candidates": topk_sorted,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if accepted:
                # update Kalman
                if consensus_used and consensus_center is not None:
                    kf.update(consensus_center)
                    last_meas_center = (float(consensus_center[0]), float(consensus_center[1]))
                else:
                    kf.update(best.center)
                    last_meas_center = (float(best.center[0]), float(best.center[1]))
                # update running radius slowly
                if seed_radius <= 1e-6:
                    seed_radius = float(best.radius)
                    prior_radius = float(seed_radius)
                else:
                    # Do not allow radius to collapse to tiny inner circles when LOST.
                    # Clamp updates around the initial calibrated radius.
                    proposed = float(0.95 * seed_radius + 0.05 * best.radius)
                    if prior_radius > 1e-6:
                        lo = float(config.get("radius_clamp_lo", 0.70)) * prior_radius
                        hi = float(config.get("radius_clamp_hi", 1.30)) * prior_radius
                        proposed = float(min(hi, max(lo, proposed)))
                    seed_radius = proposed
                # update sigmas slowly (pragmatic)
                sigma_center = float(0.95 * sigma_center + 0.05 * max(10.0, np.hypot(best.center[0] - pred_x, best.center[1] - pred_y)))
                sigma_radius = float(0.95 * sigma_radius + 0.05 * max(2.0, abs(best.radius - seed_radius)))

                # update static window (only when we actually use a measurement)
                recent_meas.append((float(best.center[0]), float(best.center[1]), float(best.radius)))

                # state
                state = "TRACKING"
                lost_counter = 0

                # emit tracked object
                det = best.det.model_copy(update={"source": "fused" if best.source in ("yolo", "geom") else best.source})
                used_sources[det.source or "unknown"] = used_sources.get(det.source or "unknown", 0) + 1
                obj = TrackedObject(track_id=master_id, detection=det, history=[])
                obj._temp_frame_idx = f  # for visualization compatibility
                disc_track.append(obj)
            else:
                lost_counter += 1
                if state == "TRACKING" and lost_counter >= lost_enter_frames:
                    state = "LOST"
                    if first_lost_frame is None:
                        first_lost_frame = int(f)
                if emit_prediction_when_lost and seed_radius > 1e-6:
                    # Emit a predicted disc so the overlay doesn't "disappear" when detections are missing.
                    # This is intentionally low-confidence and marked as "pred".
                    bbox = (
                        float(pred_x - seed_radius),
                        float(pred_y - seed_radius),
                        float(pred_x + seed_radius),
                        float(pred_y + seed_radius),
                    )
                    det = Detection(
                        class_id=disc_class_id,
                        class_name=disc_class_name,
                        confidence=0.01,
                        bbox=bbox,
                        mask=None,
                        keypoints=None,
                        source="pred",
                        radius_px=float(seed_radius),
                        shape_score=None,
                    )
                    used_sources["pred"] = used_sources.get("pred", 0) + 1
                    obj = TrackedObject(track_id=master_id, detection=det, history=[])
                    obj._temp_frame_idx = f
                    disc_track.append(obj)

                # By default we DO NOT stop early; we keep predicting so we can reacquire later.
                if stop_on_lost and state == "LOST" and lost_counter >= lost_stop_frames:
                    break

        # build output map: athlete + disc
        for f, a in athlete_ref.items():
            outputs.setdefault(f, []).append(a)
        for obj in disc_track:
            f = getattr(obj, "_temp_frame_idx", None)
            if f is None:
                continue
            outputs.setdefault(int(f), []).append(obj)

        if debug_enabled:
            with open(debug_summary, "w", encoding="utf-8") as fsum:
                json.dump(
                    {
                        "first_lost_frame": first_lost_frame,
                        "reason_counts": reason_counts,
                        "used_sources": used_sources,
                        "frames_total": len(frames_sorted),
                        "frames_tracked": len(disc_track),
                        "seed_radius_final": seed_radius,
                        "sigma_center_final": sigma_center,
                        "sigma_radius_final": sigma_radius,
                    },
                    fsum,
                    indent=2,
                )

        # cleanup temp frame idx
        for objs in outputs.values():
            for o in objs:
                if hasattr(o, "_temp_frame_idx"):
                    del o._temp_frame_idx

        return outputs

    def save_result(self, data: Dict[int, List[TrackedObject]], output_path: Path) -> None:
        serializable = {str(k): [o.model_dump() for o in v] for k, v in data.items()}
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load_result(self, input_path: Path) -> Dict[int, List[TrackedObject]]:
        with open(input_path, "r") as f:
            raw = json.load(f)
        return {int(k): [TrackedObject(**o) for o in v] for k, v in raw.items()}


