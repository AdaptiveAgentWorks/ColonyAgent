"""
Phase I: Offline Accumulation Entry Script.

Usage:
    python scripts/run_phase1.py --config configs/default.yaml --image_dir data/images --annotation_file data/annotations/train.json --output_dir memory/
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from loguru import logger

# Ensure project root is in sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Configure log output to logs/ directory
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_DIR / "phase1_{time:YYYY-MM-DD}.log", rotation="1 day", level="INFO")

from pipeline.phase1_accumulate import Phase1Pipeline


def main():
    parser = argparse.ArgumentParser(description="Phase I: Offline Accumulation")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--annotation_file", type=str, default=None,
                        help="Path to COCO format annotation JSON file")
    parser.add_argument("--output_dir", type=str, default="memory/",
                        help="Directory to save memory (experiences, skills, trajectories)")
    parser.add_argument("--num_rollouts", type=int, default=None,
                        help="Number of rollouts per image (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for processing (overrides config)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Command-line arguments override config
    if args.num_rollouts is not None:
        config.setdefault("phase1", {})["num_rollouts"] = args.num_rollouts
    if args.batch_size is not None:
        config.setdefault("phase1", {})["batch_size"] = args.batch_size

    # Run pipeline
    pipeline = Phase1Pipeline(config)

    # Log experiment settings
    logger.info("=" * 60)
    logger.info("Experiment Settings:")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Image dir: {args.image_dir}")
    logger.info(f"  Annotation file: {args.annotation_file}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Phase1 num_rollouts: {config.get('phase1', {}).get('num_rollouts', 3)}")
    logger.info(f"  Phase1 batch_size: {config.get('phase1', {}).get('batch_size', 10)}")
    logger.info(f"  Fallback enabled: {config.get('fallback', {}).get('enabled', False)}")
    logger.info(f"  Fallback search_provider: {config.get('fallback', {}).get('search_provider', 'mmx')}")
    logger.info(f"  LLM model: {config.get('llm', {}).get('model_name', 'N/A')}")
    logger.info(f"  Detector: {config.get('detectors', {}).get('yolov8', {}).get('model_path', 'N/A')}")
    logger.info("=" * 60)

    try:
        pipeline.run(args.image_dir, args.annotation_file, args.output_dir)
    except KeyboardInterrupt as e:
        logger.info(f"Pipeline stopped: {e}")
        logger.info("Memory has been saved. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
