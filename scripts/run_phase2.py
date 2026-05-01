"""
Phase II: Online Inference + Continual Learning Entry Script.

Usage:
    python scripts/run_phase2.py --config configs/default.yaml --image_dir data/images/test --memory_dir memory/ --output_dir results/
    python scripts/run_phase2.py --config configs/default.yaml --image_dir data/images/test --memory_dir memory/ --no_continual_learning
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.phase2_inference import Phase2Pipeline


def main():
    parser = argparse.ArgumentParser(description="Phase II: Online Inference + Continual Learning")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--annotation_file", type=str, default=None,
                        help="Path to COCO format annotation JSON file")
    parser.add_argument("--memory_dir", type=str, default=None,
                        help="Directory containing pre-built memory (experience_bank/, skill_library/)")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save inference results")
    parser.add_argument("--enable_continual_learning", action="store_true", default=None,
                        help="Enable continual learning during inference")
    parser.add_argument("--no_continual_learning", action="store_true", default=False,
                        help="Disable continual learning during inference")
    parser.add_argument("--detector", type=str, default=None,
                        help="Detector to use: yolov8, rtdetr, or fasterrcnn")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Command-line arguments override config
    if args.no_continual_learning:
        config.setdefault("phase2", {})["enable_continual_learning"] = False
    elif args.enable_continual_learning is not None:
        config.setdefault("phase2", {})["enable_continual_learning"] = True

    # Override detector if specified
    if args.detector:
        detector_map = {
            "yolov8": "yolov8",
            "rtdetr": "rtdetr",
            "fasterrcnn": "fasterrcnn"
        }
        det = detector_map.get(args.detector, args.detector)
        config.setdefault("detectors", {})["active"] = det

    # Override max_images if specified
    if args.max_images:
        config.setdefault("phase2", {})["max_images"] = args.max_images

    # Run pipeline
    pipeline = Phase2Pipeline(config, memory_dir=args.memory_dir)
    pipeline.run(args.image_dir, args.annotation_file, args.output_dir)


if __name__ == "__main__":
    main()
