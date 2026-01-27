import argparse
import sys
import logging
from pathlib import Path
from src.pipeline.runner import PipelineRunner

# Modules
from src.input_layer.video_loader import VideoLoader
from src.perception.yolo_segmentor import YoloSegmentor
from src.perception.yolo_pose import YoloPoseDetector
from src.perception.yolo_detector import YoloDetector
from src.visualization.video_renderer import VideoOverlayRenderer
from src.visualization.multi_model_renderer import MultiModelRenderer
from src.analysis.merger import DetectionMerger
from src.analysis.tracking.byte_tracker import ByteTracker
from src.analysis.trajectory_cleaner import TrajectoryCleaner
from src.analysis.lifting_optimizer import LiftingSessionOptimizer
from src.analysis.disc_calibrator import DiscConsensusCalibrator
from src.analysis.disc_fusion_tracker import DiscFusionTracker

def main():
    parser = argparse.ArgumentParser(description="Run the AI Pipeline")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    runner = PipelineRunner(config_path)
    
    # --- REGISTRY ---
    # Register all available modules here
    runner.register_step("video_loader", VideoLoader)
    runner.register_step("yolo_segmentor", YoloSegmentor)
    runner.register_step("yolo_pose", YoloPoseDetector)
    runner.register_step("yolo_detector", YoloDetector)  # Generic YOLO detector
    runner.register_step("video_renderer", VideoOverlayRenderer)
    runner.register_step("multi_model_renderer", MultiModelRenderer)
    runner.register_step("detection_merger", DetectionMerger)
    runner.register_step("byte_tracker", ByteTracker)
    runner.register_step("trajectory_cleaner", TrajectoryCleaner)
    runner.register_step("lifting_optimizer", LiftingSessionOptimizer)
    runner.register_step("disc_calibrator", DiscConsensusCalibrator)
    runner.register_step("disc_fusion_tracker", DiscFusionTracker)
    # ----------------
    
    try:
        runner.run()
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
