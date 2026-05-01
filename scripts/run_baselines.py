"""
Systematically run all baseline methods and collect results.

Usage:
    # Run all methods on all detectors
    python scripts/run_baselines.py --config configs/experiment.yaml

    # Run specific methods only
    python scripts/run_baselines.py --config configs/experiment.yaml --methods M0_clean M1_no_preprocess

    # Use specific detectors only
    python scripts/run_baselines.py --config configs/experiment.yaml --detectors yolov8
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

# Configure loguru to save to file
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(os.path.join(LOG_DIR, f"baselines_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"), rotation="1 day", level="INFO")

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tools.registry import get_tool
from utils.metrics import (
    calculate_map,
    calculate_recall,
    calculate_f1,
    batch_colony_metrics,
)

# ---------------------------------------------------------------------------
# Import all tool modules so @register_tool decorators execute
# ---------------------------------------------------------------------------
import tools.preprocessing.clahe  # noqa: F401
import tools.preprocessing.denoise  # noqa: F401
import tools.preprocessing.illumination  # noqa: F401
import tools.preprocessing.roi_extract  # noqa: F401
import tools.preprocessing.sharpen  # noqa: F401
import tools.preprocessing.white_balance  # noqa: F401
import tools.preprocessing.colony_separation  # noqa: F401
import tools.detection.yolov8_det  # noqa: F401
import tools.detection.rtdetr_det  # noqa: F401
import tools.detection.fasterrcnn_det  # noqa: F401


# ---------------------------------------------------------------------------
# YOLO label loading
# ---------------------------------------------------------------------------

def load_yolo_labels(label_dir: str, img_width: int, img_height: int) -> Dict[str, List[Dict]]:
    """
    Load YOLO txt format annotations, convert to {filename: List[Dict]} format.

    YOLO format: class_id x_center y_center width height (normalized 0-1)
    Convert to xyxy pixel coordinates.

    Args:
        label_dir: Annotation file directory
        img_width: Image width (for denormalization)
        img_height: Image height (for denormalization)

    Returns:
        {label_filename_stem: [{"bbox": [x1,y1,x2,y2], "class_id": int}, ...]}
    """
    annotations = {}
    if not os.path.isdir(label_dir):
        logger.warning(f"Label directory not found: {label_dir}")
        return annotations

    for label_file in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(label_file))[0]
        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                xc = float(parts[1]) * img_width
                yc = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height
                x1 = xc - w / 2
                y1 = yc - h / 2
                x2 = xc + w / 2
                y2 = yc + h / 2
                boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                })
        annotations[stem] = boxes
    return annotations


def load_yolo_labels_for_image(label_path: str, img_width: int, img_height: int) -> List[Dict]:
    """Load YOLO labels for a single image."""
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            xc = float(parts[1]) * img_width
            yc = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
            })
    return boxes


# ---------------------------------------------------------------------------
# Detector name → tool name mapping
# ---------------------------------------------------------------------------

DETECTOR_TOOL_MAP = {
    "yolov8": "yolov8_detect",
    "rtdetr": "rtdetr_detect",
    "fasterrcnn": "fasterrcnn_detect",
}


# ---------------------------------------------------------------------------
# Run a single method × detector combination
# ---------------------------------------------------------------------------

def run_method_detector(
    method_name: str,
    method_cfg: dict,
    detector_name: str,
    detector_cfg,
    image_dir: str,
    label_dir: str,
    output_dir: str,
    default_config_path: str = "configs/default.yaml",
    batch_size: int = 32,
) -> Dict:
    """
    Execute processing and detection for all images for a given method × detector combination.

    Uses batch processing for faster inference.

    Returns:
        Aggregated results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve detector tool
    tool_name = DETECTOR_TOOL_MAP.get(detector_name)
    if tool_name is None:
        logger.error(f"Unknown detector: {detector_name}")
        return {}

    detector_tool = get_tool(tool_name)

    # Resolve preprocessing tools for M2
    preprocess_tools = []
    if method_cfg.get("preprocess") == "fixed":
        fixed_tools_cfg = method_cfg.get("fixed_tools", [])
        for tc in fixed_tools_cfg:
            preprocess_tools.append((get_tool(tc["name"]), tc.get("params", {})))

    # Collect image files
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files = sorted(image_files)

    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return {}

    logger.info(
        f"Running {method_name} + {detector_name}: "
        f"{len(image_files)} images from {image_dir} (batch_size={batch_size})"
    )

    per_image_results = []
    colony_count_records = []
    total_time = 0.0

    # Use batch processing for YOLOv8 and RT-DETR (Faster R-CNN has different API)
    use_batch = detector_name in ["yolov8", "rtdetr"]
    
    if use_batch:
        # Batch processing for YOLOv8/RT-DETR
        for batch_start in range(0, len(image_files), batch_size):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]
            
            logger.info(f"  Processing batch {batch_start//batch_size + 1}/{(len(image_files)-1)//batch_size + 1}: images {batch_start}-{batch_end}")
            
            # Load batch images
            batch_images = []
            batch_metadata = []  # Store metadata for each image
            
            for img_path in batch_files:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, f"{stem}.txt")
                
                image = cv2.imread(img_path)
                if image is None:
                    per_image_results.append({
                        "image_name": os.path.basename(img_path),
                        "error": "Failed to load image",
                    })
                    continue
                
                img_h, img_w = image.shape[:2]
                gt_boxes = load_yolo_labels_for_image(label_path, img_w, img_h)
                
                # Apply preprocessing
                processed = image.copy()
                preprocess_log = []
                for tool, params in preprocess_tools:
                    result = tool.call(processed, **params)
                    if result.success:
                        processed = result.image
                        preprocess_log.append({"tool": tool.name, "params": params, "success": True})
                    else:
                        preprocess_log.append({"tool": tool.name, "params": params, "success": False, "error": result.error})
                
                batch_images.append(processed)
                batch_metadata.append({
                    "img_path": img_path,
                    "gt_boxes": gt_boxes,
                    "preprocess_log": preprocess_log,
                })
            
            if not batch_images:
                continue
            
            t0 = time.time()
            
            # Batch detection
            model_path = detector_cfg if isinstance(detector_cfg, str) else detector_cfg.get("model_path", "")
            det_kwargs = {
                "model_path": model_path,
                "conf_threshold": 0.5,
                "iou_threshold": 0.45,
                "device": "cuda:1",  # Explicitly use GPU
                "batch_images": batch_images,  # Pass batch to detector
            }
            # Pass num_classes if specified in config (for fasterrcnn)
            if isinstance(detector_cfg, dict) and "num_classes" in detector_cfg:
                det_kwargs["num_classes"] = detector_cfg["num_classes"]
            det_result = detector_tool.call(batch_images[0], **det_kwargs)
            
            elapsed = time.time() - t0
            total_time += elapsed
            
            # Process results per image
            detections_list = det_result.detections if det_result.success else []
            
            for i, meta in enumerate(batch_metadata):
                detections = detections_list[i] if i < len(detections_list) else []
                gt_boxes = meta["gt_boxes"]
                
                if gt_boxes:
                    img_map_result = calculate_map(detections, gt_boxes, [0.5])
                    img_map = img_map_result.get("mAP", 0.0)
                    img_recall = calculate_recall(detections, gt_boxes, 0.5)
                else:
                    img_map = 0.0
                    img_recall = 0.0
                
                pred_count = len(detections)
                gt_count = len(gt_boxes)
                
                record = {
                    "image_name": os.path.basename(meta["img_path"]),
                    "pred_count": pred_count,
                    "gt_count": gt_count,
                    "mAP@0.5": img_map,
                    "recall": img_recall,
                    "processing_time": elapsed / len(batch_metadata),  # Per-image time
                    "detections": detections,
                    "preprocess_log": meta["preprocess_log"],
                }
                per_image_results.append(record)
                colony_count_records.append({"pred_count": pred_count, "gt_count": gt_count})
    else:
        # Original single-image processing for Faster R-CNN
        for img_path in image_files:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{stem}.txt")

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                per_image_results.append({
                    "image_name": os.path.basename(img_path),
                    "error": "Failed to load image",
                })
                continue

            img_h, img_w = image.shape[:2]

            # Load GT
            gt_boxes = load_yolo_labels_for_image(label_path, img_w, img_h)

            t0 = time.time()

            # Apply preprocessing (M2 fixed pipeline)
            processed = image.copy()
            preprocess_log = []
            for tool, params in preprocess_tools:
                result = tool.call(processed, **params)
                if result.success:
                    processed = result.image
                    preprocess_log.append({
                        "tool": tool.name,
                        "params": params,
                        "success": True,
                    })
                else:
                    preprocess_log.append({
                        "tool": tool.name,
                        "params": params,
                        "success": False,
                        "error": result.error,
                    })

            # Detect
            model_path = detector_cfg if isinstance(detector_cfg, str) else detector_cfg.get("model_path", "")
            det_kwargs = {
                "model_path": model_path,
                "conf_threshold": 0.5,
                "iou_threshold": 0.45,
                "device": "cuda:1",  # Explicitly use GPU
            }
            # Faster R-CNN requires specifying num_classes
            if isinstance(detector_cfg, dict) and "num_classes" in detector_cfg:
                det_kwargs["num_classes"] = detector_cfg["num_classes"]
            elif detector_name == "fasterrcnn":
                det_kwargs["num_classes"] = 8
            det_result = detector_tool.call(processed, **det_kwargs)

            elapsed = time.time() - t0
            total_time += elapsed

            detections = det_result.detections if det_result.success else []

            # Per-image metrics
            if gt_boxes:
                img_map_result = calculate_map(detections, gt_boxes, [0.5])
                img_map = img_map_result.get("mAP", 0.0)
                img_recall = calculate_recall(detections, gt_boxes, 0.5)
            else:
                img_map = 0.0
                img_recall = 0.0

            pred_count = len(detections)
            gt_count = len(gt_boxes)

            record = {
                "image_name": os.path.basename(img_path),
                "pred_count": pred_count,
                "gt_count": gt_count,
                "mAP@0.5": img_map,
                "recall": img_recall,
                "processing_time": elapsed,
                "detections": detections,
                "preprocess_log": preprocess_log,
            }
            per_image_results.append(record)

            colony_count_records.append({
                "pred_count": pred_count,
                "gt_count": gt_count,
            })

    # Aggregate
    valid = [r for r in per_image_results if "error" not in r]
    if valid:
        avg_map = float(np.mean([r["mAP@0.5"] for r in valid]))
        avg_recall = float(np.mean([r["recall"] for r in valid]))
        avg_f1 = calculate_f1(avg_map, avg_recall)
        colony_metrics = batch_colony_metrics(colony_count_records)
    else:
        avg_map = 0.0
        avg_recall = 0.0
        avg_f1 = 0.0
        colony_metrics = {"mae": 0.0, "rmse": 0.0, "avg_recovery_rate": 0.0}

    summary = {
        "method": method_name,
        "detector": detector_name,
        "description": method_cfg.get("description", ""),
        "num_images": len(image_files),
        "num_valid": len(valid),
        "avg_mAP@0.5": avg_map,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "colony_count_mae": colony_metrics["mae"],
        "colony_count_rmse": colony_metrics["rmse"],
        "avg_recovery_rate": colony_metrics["avg_recovery_rate"],
        "total_time_sec": total_time,
        "avg_time_per_image": total_time / len(image_files) if image_files else 0.0,
    }

    # Save per-image results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        f"  {method_name}+{detector_name}: "
        f"mAP={avg_map:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}, "
        f"MAE={colony_metrics['mae']:.2f}, "
        f"Recovery={colony_metrics['avg_recovery_rate']:.1f}%"
    )

    # Final memory save for continual learning (M4)
    if method_cfg.get("preprocess") == "agent" and method_name.startswith("M4"):
        pipeline.flush_memory()

    return summary


# ---------------------------------------------------------------------------
# Run agent-based methods (M3/M4) via Phase2Pipeline
# ---------------------------------------------------------------------------

def run_agent_method(
    method_name: str,
    method_cfg: dict,
    detector_name: str,
    detector_cfg,
    image_dir: str,
    label_dir: str,
    output_dir: str,
    default_config_path: str = "configs/default.yaml",
    memory_dir: str = "memory/",
) -> Dict:
    """
    Run Agent methods (M3/M4) via Phase2Pipeline.

    detector_cfg: either a string (model_path) or a dict with model_path and other params like num_classes
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load default config
    with open(default_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override continual learning setting
    if "phase2" not in config:
        config["phase2"] = {}
    config["phase2"]["enable_continual_learning"] = method_cfg.get(
        "continual_learning", False
    )

    # Override detector - support both string and dict config formats
    if "detectors" not in config:
        config["detectors"] = {}
    config["detectors"]["active"] = detector_name
    if isinstance(detector_cfg, dict):
        # Merge full config dict to preserve num_classes etc.
        existing = config["detectors"].get(detector_name, {})
        config["detectors"][detector_name] = {**existing, **detector_cfg}
    else:
        # String path format
        config["detectors"][detector_name] = {"model_path": detector_cfg}

    try:
        from pipeline.phase2_inference import Phase2Pipeline

        pipeline = Phase2Pipeline(config=config, memory_dir=memory_dir)
    except Exception as e:
        logger.error(f"Failed to initialize Phase2Pipeline: {e}")
        return {"method": method_name, "detector": detector_name, "error": str(e)}

    # Collect image files
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files = sorted(image_files)

    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return {}

    logger.info(
        f"Running {method_name} + {detector_name} (agent): "
        f"{len(image_files)} images"
    )

    per_image_results = []
    colony_count_records = []
    total_time = 0.0

    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{stem}.txt")

        image = cv2.imread(img_path)
        if image is None:
            per_image_results.append({
                "image_name": os.path.basename(img_path),
                "error": "Failed to load image",
            })
            continue

        img_h, img_w = image.shape[:2]
        gt_boxes = load_yolo_labels_for_image(label_path, img_w, img_h)

        t0 = time.time()
        try:
            # Pass annotations for M3/M4 comparison logic
            result = pipeline.process_single(image, annotations=gt_boxes)
            elapsed = time.time() - t0
            total_time += elapsed

            detections = result.get("detections", [])
            pred_count = len(detections)
            gt_count = len(gt_boxes)

            if gt_boxes:
                img_map_result = calculate_map(detections, gt_boxes, [0.5])
                img_map = img_map_result.get("mAP", 0.0)
                img_recall = calculate_recall(detections, gt_boxes, 0.5)
            else:
                img_map = 0.0
                img_recall = 0.0

            record = {
                "image_name": os.path.basename(img_path),
                "pred_count": pred_count,
                "gt_count": gt_count,
                "mAP@0.5": img_map,
                "recall": img_recall,
                "processing_time": elapsed,
                "detections": detections,
                "tools_used": result.get("tools_used", []),
                # M3/M4 comparison fields
                "detection_selection": result.get("detection_selection", "N/A"),
                "direct_vs_experience_mAP_delta": result.get("direct_vs_experience_mAP_delta", 0.0),
                "why_experience_worse": result.get("why_experience_worse", None),
                "direct_detections_count": result.get("direct_detections_count", 0),
                "experience_detections_count": result.get("experience_detections_count", 0),
            }
            per_image_results.append(record)
            colony_count_records.append({
                "pred_count": pred_count,
                "gt_count": gt_count,
            })

        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            logger.error(f"Agent processing failed for {img_path}: {e}")
            per_image_results.append({
                "image_name": os.path.basename(img_path),
                "error": str(e),
            })

    # Aggregate
    valid = [r for r in per_image_results if "error" not in r]
    if valid:
        avg_map = float(np.mean([r["mAP@0.5"] for r in valid]))
        avg_recall = float(np.mean([r["recall"] for r in valid]))
        avg_f1 = calculate_f1(avg_map, avg_recall)
        colony_metrics = batch_colony_metrics(colony_count_records)
    else:
        avg_map = 0.0
        avg_recall = 0.0
        avg_f1 = 0.0
        colony_metrics = {"mae": 0.0, "rmse": 0.0, "avg_recovery_rate": 0.0}

    summary = {
        "method": method_name,
        "detector": detector_name,
        "description": method_cfg.get("description", ""),
        "num_images": len(image_files),
        "num_valid": len(valid),
        "avg_mAP@0.5": avg_map,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "colony_count_mae": colony_metrics["mae"],
        "colony_count_rmse": colony_metrics["rmse"],
        "avg_recovery_rate": colony_metrics["avg_recovery_rate"],
        "total_time_sec": total_time,
        "avg_time_per_image": total_time / len(image_files) if image_files else 0.0,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        f"  {method_name}+{detector_name}: "
        f"mAP={avg_map:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}, "
        f"MAE={colony_metrics['mae']:.2f}, "
        f"Recovery={colony_metrics['avg_recovery_rate']:.1f}%"
    )

    # Final memory save for M4 (continual learning)
    if method_cfg.get("continual_learning", False):
        pipeline.flush_memory()

    return summary


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_summaries: List[Dict]):
    """Print summary of all methods in table format."""
    if not all_summaries:
        logger.warning("No summaries to print")
        return

    header = (
        f"{'Method':<20} {'Detector':<12} {'mAP@0.5':>8} {'Recall':>8} "
        f"{'F1':>8} {'MAE':>8} {'Recovery%':>10} {'Time/img':>10}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("EXPERIMENT RESULTS SUMMARY")
    print(separator)
    print(header)
    print(separator)

    for s in all_summaries:
        if "error" in s:
            print(f"{s.get('method', '?'):<20} {s.get('detector', '?'):<12} ERROR: {s['error']}")
            continue
        print(
            f"{s.get('method', '?'):<20} {s.get('detector', '?'):<12} "
            f"{s.get('avg_mAP@0.5', 0):>8.4f} {s.get('avg_recall', 0):>8.4f} "
            f"{s.get('avg_f1', 0):>8.4f} {s.get('colony_count_mae', 0):>8.2f} "
            f"{s.get('avg_recovery_rate', 0):>10.1f} "
            f"{s.get('avg_time_per_image', 0):>10.4f}s"
        )

    print(separator + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run all baseline experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Experiment config path",
    )
    parser.add_argument(
        "--default_config",
        type=str,
        default="configs/default.yaml",
        help="Default pipeline config path (for agent methods)",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Methods to run (e.g. M0_clean M1_no_preprocess). Default: all.",
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        default=None,
        help="Detectors to use (e.g. yolov8 rtdetr). Default: all.",
    )
    parser.add_argument(
        "--memory_dir",
        type=str,
        default="memory/",
        help="Memory directory for agent methods",
    )
    args = parser.parse_args()

    # Load experiment config
    with open(args.config, "r", encoding="utf-8") as f:
        exp_config = yaml.safe_load(f)

    experiment = exp_config["experiment"]
    data_cfg = experiment["data"]
    detectors_cfg = experiment["detectors"]
    methods_cfg = experiment["methods"]
    output_base = experiment.get("output_dir", "results/")

    # Filter methods and detectors
    method_names = args.methods or list(methods_cfg.keys())
    detector_names = args.detectors or list(detectors_cfg.keys())

    logger.info(f"Methods to run: {method_names}")
    logger.info(f"Detectors to use: {detector_names}")

    all_summaries = []

    for method_name in method_names:
        if method_name not in methods_cfg:
            logger.warning(f"Method {method_name} not found in config, skipping")
            continue

        method_cfg = methods_cfg[method_name]

        for detector_name in detector_names:
            if detector_name not in detectors_cfg:
                logger.warning(f"Detector {detector_name} not found in config, skipping")
                continue

            # Support both string and object config formats
            detector_cfg = detectors_cfg[detector_name]
            output_dir = os.path.join(output_base, f"{method_name}_{detector_name}")

            # Determine image and label directories
            if method_cfg.get("use_clean", False):
                # M0: use clean validation images
                image_dir = data_cfg["clean_val_images"]
                label_dir = data_cfg["clean_val_labels"]
            else:
                # M1/M2/M3/M4: use degraded test images
                image_dir = data_cfg["degraded_test"]
                label_dir = os.path.join(data_cfg["degraded_test"], "labels")
                # If degraded_test has images/ subdir
                if os.path.isdir(os.path.join(data_cfg["degraded_test"], "images")):
                    image_dir = os.path.join(data_cfg["degraded_test"], "images")

            preprocess_mode = method_cfg.get("preprocess", "none")

            if preprocess_mode == "agent":
                # M3/M4: use Phase2Pipeline
                # For M4 (continual learning), each run should use independent memory
                memory_dir = args.memory_dir
                if method_name.startswith("M4") and method_cfg.get("continual_learning", False):
                    # Create unique memory dir for each M4 detector run
                    unique_memory = f"memory_m4_{detector_name}_{int(time.time())}"
                    import shutil
                    if os.path.exists(memory_dir):
                        shutil.copytree(memory_dir, unique_memory)
                    memory_dir = unique_memory
                    logger.info(f"M4 using independent memory: {memory_dir}")

                summary = run_agent_method(
                    method_name=method_name,
                    method_cfg=method_cfg,
                    detector_name=detector_name,
                    detector_cfg=detector_cfg,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    output_dir=output_dir,
                    default_config_path=args.default_config,
                    memory_dir=memory_dir,
                )
            else:
                # M0/M1/M2: direct tool calls
                summary = run_method_detector(
                    method_name=method_name,
                    method_cfg=method_cfg,
                    detector_name=detector_name,
                    detector_cfg=detector_cfg,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    output_dir=output_dir,
                    default_config_path=args.default_config,
                )

            all_summaries.append(summary)

    # Save overall summary
    overall_path = os.path.join(output_base, "all_summaries.json")
    os.makedirs(output_base, exist_ok=True)
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    logger.info(f"Overall summary saved to {overall_path}")

    # Print table
    print_summary_table(all_summaries)


if __name__ == "__main__":
    main()
