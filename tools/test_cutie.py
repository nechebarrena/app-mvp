#!/usr/bin/env python3
"""
Cutie Model Validation Tool

Standalone script to test the Cutie VOS tracker on a video with disc selection.
Produces annotated frames and a summary report to validate tracking quality.

Usage:
    # From ai-core directory:
    PYTHONPATH=vendors/cutie:src:. uv run python ../tools/test_cutie.py <video_path> <selection_json>
    
    # Or with explicit selection:
    PYTHONPATH=vendors/cutie:src:. uv run python ../tools/test_cutie.py <video_path> --center 375 1158 --radius 109
    
    # Save annotated output video:
    PYTHONPATH=vendors/cutie:src:. uv run python ../tools/test_cutie.py <video_path> <selection_json> --save-video

Examples:
    # Test with the problematic video:
    cd ai-core
    PYTHONPATH=vendors/cutie:src:. uv run python ../tools/test_cutie.py \\
        ../data/api/uploads/518cc0b7-fe5/input.mp4 \\
        ../data/api/uploads/518cc0b7-fe5/disc_selection.json \\
        --save-video
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor


def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cutie_model(weights_path: str, device: torch.device,
                     max_internal_size: int = 480, mem_every: int = 5):
    """Load Cutie model and create inference processor."""
    from omegaconf import open_dict
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from cutie.model.cutie import CUTIE
    from cutie.inference.inference_core import InferenceCore

    GlobalHydra.instance().clear()

    # Find config dir relative to cutie package
    import cutie
    cutie_pkg_dir = Path(cutie.__path__[0])
    config_dir = str(cutie_pkg_dir / "config")

    with initialize_config_dir(version_base='1.3.2', config_dir=config_dir):
        cfg = compose(config_name="eval_config")

    with open_dict(cfg):
        cfg['weights'] = weights_path
        cfg['max_internal_size'] = max_internal_size
        cfg['mem_every'] = mem_every
        cfg['flip_aug'] = False
        cfg['save_aux'] = False
        cfg['chunk_size'] = -1
        cfg['stagger_updates'] = min(5, mem_every)

    model = CUTIE(cfg).to(device).eval()
    model_weights = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_weights(model_weights)

    processor = InferenceCore(model, cfg=cfg)
    processor.max_internal_size = max_internal_size

    return model, processor


def create_circular_mask(height, width, center, radius):
    """Create binary mask from disc selection."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 1, -1)
    return mask


def annotate_frame(frame_bgr, mask_np, frame_idx, confidence, color=(0, 255, 0)):
    """Draw mask overlay, bbox, and info on frame."""
    annotated = frame_bgr.copy()
    
    binary = (mask_np == 1).astype(np.uint8)
    area = binary.sum()
    
    if area > 0:
        # Semi-transparent mask overlay
        overlay = annotated.copy()
        overlay[binary == 1] = color
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
        
        # Contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, 2)
        
        # Bbox
        ys, xs = np.where(binary > 0)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        
        # Center dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        
        # Info text
        label = f"F{frame_idx} conf={confidence:.2f} area={area}"
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        cv2.putText(annotated, f"F{frame_idx} NO DETECTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Test Cutie VOS tracker on a video with disc selection"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("selection", nargs="?", help="Path to disc_selection.json")
    parser.add_argument("--center", type=float, nargs=2, help="Disc center (x y)")
    parser.add_argument("--radius", type=float, help="Disc radius in pixels")
    parser.add_argument("--weights", default="models/pretrained/cutie/cutie-base-mega.pth",
                        help="Path to Cutie weights")
    parser.add_argument("--max-size", type=int, default=480,
                        help="Max internal processing size (default: 480)")
    parser.add_argument("--mem-every", type=int, default=5,
                        help="Store memory every N frames (default: 5)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated output video")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="Directory to save annotated key frames")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: alongside video)")
    args = parser.parse_args()

    # --- Resolve selection data ---
    if args.selection:
        with open(args.selection) as f:
            sel = json.load(f)
        center = sel["center"]
        radius = sel["radius"]
    elif args.center and args.radius:
        center = args.center
        radius = args.radius
    else:
        parser.error("Must provide either selection JSON or --center and --radius")

    print(f"{'='*60}")
    print(f"  Cutie VOS Tracker - Validation Tool")
    print(f"{'='*60}")
    print(f"Video:     {args.video}")
    print(f"Selection: center=({center[0]:.0f}, {center[1]:.0f}), radius={radius:.0f}")
    print(f"Weights:   {args.weights}")
    print(f"Max size:  {args.max_size}")
    print(f"Mem every: {args.mem_every}")
    print()

    # --- Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info: {w}x{h}, {fps:.1f} fps, {total_frames} frames, {total_frames/fps:.1f}s")
    print()

    # --- Load model ---
    device = get_device()
    print(f"Loading Cutie model on {device}...")
    t0 = time.time()
    model, processor = load_cutie_model(args.weights, device, args.max_size, args.mem_every)
    print(f"Model loaded in {time.time()-t0:.1f}s")
    print()

    # --- Create initial mask ---
    initial_mask = create_circular_mask(h, w, center, radius)
    initial_mask_tensor = torch.from_numpy(initial_mask).long().to(device)

    # --- Output setup ---
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.video).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if args.save_video:
        out_path = output_dir / f"cutie_result_{Path(args.video).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"Saving annotated video to: {out_path}")

    if args.save_frames:
        frames_dir = Path(args.save_frames)
        frames_dir.mkdir(parents=True, exist_ok=True)

    # --- Process frames ---
    print(f"\nProcessing {total_frames} frames...")
    frame_idx = 0
    start_ts = time.time()
    
    # Tracking statistics
    areas = []
    centers_x = []
    centers_y = []
    confidences = []
    empty_frames = []

    with torch.inference_mode():
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = to_tensor(frame_rgb).float().to(device)

            if frame_idx == 0:
                output_prob = processor.step(image_tensor, initial_mask_tensor, objects=[1])
            else:
                output_prob = processor.step(image_tensor)

            output_mask = processor.output_prob_to_mask(output_prob)
            mask_np = output_mask.cpu().numpy().astype(np.uint8)

            # Confidence from probability
            binary = (mask_np == 1).astype(np.uint8)
            area = int(binary.sum())

            if output_prob.shape[0] > 1 and area > 0:
                obj_prob = output_prob[1]
                confidence = float(obj_prob[binary == 1].mean().cpu())
            else:
                confidence = 0.0

            # Collect stats
            if area > 100:  # min meaningful area
                ys, xs = np.where(binary > 0)
                cx = float(xs.mean())
                cy = float(ys.mean())
                areas.append(area)
                centers_x.append(cx)
                centers_y.append(cy)
                confidences.append(confidence)
            else:
                empty_frames.append(frame_idx)

            # Annotate
            annotated = annotate_frame(frame_bgr, mask_np, frame_idx, confidence)

            if video_writer:
                video_writer.write(annotated)

            if args.save_frames and (frame_idx % 30 == 0 or frame_idx < 5):
                cv2.imwrite(str(Path(args.save_frames) / f"frame_{frame_idx:04d}.jpg"), annotated)

            frame_idx += 1

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_ts
                fps_proc = frame_idx / max(1e-6, elapsed)
                pct = 100.0 * frame_idx / total_frames
                eta = (total_frames - frame_idx) / max(1e-6, fps_proc)
                print(f"  [{frame_idx}/{total_frames}] {pct:.0f}% | {fps_proc:.1f} fps | ETA {eta:.0f}s", end="\r")

    cap.release()
    if video_writer:
        video_writer.release()

    elapsed_total = time.time() - start_ts
    fps_avg = frame_idx / max(1e-6, elapsed_total)

    # --- Summary report ---
    tracked_frames = frame_idx - len(empty_frames)
    coverage = tracked_frames / max(1, frame_idx) * 100

    print(f"\n\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Processing time:    {elapsed_total:.1f}s ({fps_avg:.1f} fps)")
    print(f"Total frames:       {frame_idx}")
    print(f"Tracked frames:     {tracked_frames} ({coverage:.1f}% coverage)")
    print(f"Empty frames:       {len(empty_frames)}")
    
    if areas:
        print(f"\nMask area (px):     min={min(areas)}, max={max(areas)}, avg={np.mean(areas):.0f}")
        print(f"Confidence:         min={min(confidences):.3f}, max={max(confidences):.3f}, avg={np.mean(confidences):.3f}")
        
        # Trajectory summary
        print(f"\nTrajectory:")
        print(f"  X range: {min(centers_x):.0f} - {max(centers_x):.0f} px")
        print(f"  Y range: {min(centers_y):.0f} - {max(centers_y):.0f} px")
        total_y_displacement = max(centers_y) - min(centers_y)
        print(f"  Total Y displacement: {total_y_displacement:.0f} px")
    
    if empty_frames:
        if len(empty_frames) <= 20:
            print(f"\nEmpty frame indices: {empty_frames}")
        else:
            print(f"\nFirst 10 empty frames: {empty_frames[:10]}")
            print(f"Last 10 empty frames: {empty_frames[-10:]}")
    
    # Compare with YOLO results if available
    results_dir = Path(args.video).parent.parent / "results" / Path(args.video).parent.name
    results_json = results_dir / "results.json"
    if results_json.exists():
        print(f"\n{'='*60}")
        print(f"  COMPARISON WITH YOLO RESULTS")
        print(f"{'='*60}")
        with open(results_json) as f:
            yolo_results = json.load(f)
        for track in yolo_results.get("tracks", []):
            if track["class_name"] in ("frisbee", "sports ball"):
                n_frames = len(track["frames"])
                confs = [track["frames"][f]["confidence"] for f in track["frames"]]
                print(f"  YOLO Track {track['track_id']} ({track['class_name']}): "
                      f"{n_frames} frames, conf={min(confs):.3f}-{max(confs):.3f}")
        print(f"  Cutie: {tracked_frames} frames, conf={min(confidences):.3f}-{max(confidences):.3f}")
        print(f"  Improvement: {tracked_frames - 28}+ more frames tracked")
    
    # Save summary JSON
    summary = {
        "video": str(args.video),
        "selection": {"center": center, "radius": radius},
        "processing": {
            "device": str(device),
            "max_internal_size": args.max_size,
            "mem_every": args.mem_every,
            "elapsed_s": round(elapsed_total, 2),
            "fps": round(fps_avg, 2),
        },
        "results": {
            "total_frames": frame_idx,
            "tracked_frames": tracked_frames,
            "coverage_pct": round(coverage, 2),
            "empty_frames": len(empty_frames),
            "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0,
            "avg_mask_area": round(float(np.mean(areas)), 0) if areas else 0,
        }
    }
    summary_path = output_dir / f"cutie_summary_{Path(args.video).stem}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    if args.save_video:
        print(f"Annotated video saved to: {output_dir / f'cutie_result_{Path(args.video).stem}.mp4'}")

    print()


if __name__ == "__main__":
    main()
