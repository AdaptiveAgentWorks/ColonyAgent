"""Evaluation module - Evaluates detection performance, compares methods, and analyzes learning curves."""

import json
import os
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from utils.metrics import calculate_map, calculate_recall, calculate_f1


class Evaluator:
    """Evaluates detection performance."""

    @staticmethod
    def evaluate_results(results_dir: str, annotation_file: str) -> Dict:
        """
        Load all results from result directory, calculate overall metrics:
        - Overall mAP@0.5, mAP@0.5:0.95
        - Overall Recall, Precision, F1
        - Preprocessing gain (average mAP delta after vs before processing)
        - Performance grouped by image quality (low/medium/high)
        """
        # Load results
        results_path = os.path.join(results_dir, "results.json")
        if not os.path.exists(results_path):
            logger.error(f"Results file not found: {results_path}")
            return {}

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Load annotations
        annotations = Evaluator._load_annotations(annotation_file)

        if not results or not annotations:
            logger.warning("No results or annotations to evaluate")
            return {}

        # Aggregate detections and ground truths
        all_preds = []
        all_gts = []
        mAP_deltas = []

        quality_groups = {"low": [], "medium": [], "high": []}

        for result in results:
            if "error" in result:
                continue

            img_name = result.get("image_name", "")
            gt_anns = annotations.get(img_name, [])
            detections = result.get("detections", [])

            # Accumulate for overall metrics
            all_preds.extend(detections)
            all_gts.extend(gt_anns)

            # Per-image mAP
            if gt_anns:
                img_map_result = calculate_map(detections, gt_anns, [0.5])
                img_map = img_map_result.get("mAP", 0.0)
            else:
                img_map = 0.0

            # mAP delta from feedback
            feedback = result.get("feedback", {})
            delta = feedback.get("delta", {})
            if "mAP" in delta:
                mAP_deltas.append(delta["mAP"])

            # Group by quality
            quality_report = result.get("quality_report", {})
            overall_quality = quality_report.get("overall_score", 0.5)

            if overall_quality < 0.4:
                quality_groups["low"].append(img_map)
            elif overall_quality < 0.7:
                quality_groups["medium"].append(img_map)
            else:
                quality_groups["high"].append(img_map)

        # Overall metrics
        iou_thresholds_50 = [0.5]
        iou_thresholds_coco = [0.5 + i * 0.05 for i in range(10)]  # 0.5:0.05:0.95

        map_50_result = calculate_map(all_preds, all_gts, iou_thresholds_50)
        map_coco_result = calculate_map(all_preds, all_gts, iou_thresholds_coco)

        overall_recall = calculate_recall(all_preds, all_gts, 0.5)

        # Precision: approximate from mAP@0.5 (using AP as proxy)
        overall_precision = map_50_result.get("mAP", 0.0)
        overall_f1 = calculate_f1(overall_precision, overall_recall)

        # Preprocessing gain
        avg_mAP_delta = float(np.mean(mAP_deltas)) if mAP_deltas else 0.0

        # Quality group performance
        quality_performance = {}
        for group, maps in quality_groups.items():
            if maps:
                quality_performance[group] = {
                    "count": len(maps),
                    "mean_mAP": float(np.mean(maps)),
                    "std_mAP": float(np.std(maps)),
                }
            else:
                quality_performance[group] = {
                    "count": 0,
                    "mean_mAP": 0.0,
                    "std_mAP": 0.0,
                }

        evaluation = {
            "num_images": len([r for r in results if "error" not in r]),
            "mAP@0.5": map_50_result.get("mAP", 0.0),
            "mAP@0.5:0.95": map_coco_result.get("mAP", 0.0),
            "recall": overall_recall,
            "precision": overall_precision,
            "f1": overall_f1,
            "preprocessing_gain_mAP": avg_mAP_delta,
            "quality_group_performance": quality_performance,
            "per_class_ap": map_50_result.get("per_class", {}),
        }

        logger.info(
            f"Evaluation: mAP@0.5={evaluation['mAP@0.5']:.4f}, "
            f"mAP@0.5:0.95={evaluation['mAP@0.5:0.95']:.4f}, "
            f"F1={evaluation['f1']:.4f}, "
            f"preprocessing_gain={avg_mAP_delta:+.4f}"
        )

        return evaluation

    @staticmethod
    def compare_methods(results_dirs: Dict[str, str], annotation_file: str) -> Dict:
        """
        Compare results of multiple methods.

        Args:
            results_dirs: {"method_name": "results_path", ...}
                e.g., {"no_preprocess": "path1", "fixed_pipeline": "path2", "our_method": "path3"}
            annotation_file: COCO format annotation file path.

        Returns:
            {method_name: evaluation_dict, ...} + "comparison" key containing comparison information.
        """
        method_results = {}

        for method_name, results_dir in results_dirs.items():
            logger.info(f"Evaluating method: {method_name}")
            evaluation = Evaluator.evaluate_results(results_dir, annotation_file)
            method_results[method_name] = evaluation

        # Generate comparison
        comparison = {
            "methods": list(results_dirs.keys()),
            "metrics": {},
        }

        metrics_to_compare = ["mAP@0.5", "mAP@0.5:0.95", "recall", "precision", "f1"]
        for metric in metrics_to_compare:
            values = {}
            for method_name, evaluation in method_results.items():
                values[method_name] = evaluation.get(metric, 0.0)
            comparison["metrics"][metric] = values

            # Find best
            if values:
                best_method = max(values, key=values.get)
                comparison["metrics"][metric]["best"] = best_method

        method_results["comparison"] = comparison

        logger.info(f"Compared {len(results_dirs)} methods")
        return method_results

    @staticmethod
    def plot_learning_curve(results_dir: str) -> Dict:
        """
        Analyze learning curve: changes in mAP as the number of processed images increases.

        Args:
            results_dir: Result directory path.

        Returns:
            {"num_images": [10, 20, ...], "mAP": [0.65, 0.68, ...]}
        """
        results_path = os.path.join(results_dir, "results.json")
        if not os.path.exists(results_path):
            logger.error(f"Results file not found: {results_path}")
            return {"num_images": [], "mAP": []}

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Filter valid results
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return {"num_images": [], "mAP": []}

        # Calculate cumulative mAP at different points
        step_size = max(1, len(valid_results) // 20)  # ~20 points on the curve
        num_images_list = []
        mAP_list = []
        verdict_improved_ratio = []

        for i in range(step_size, len(valid_results) + 1, step_size):
            subset = valid_results[:i]
            num_images_list.append(i)

            # Count verdicts
            improved = sum(1 for r in subset if r.get("verdict") == "improved")
            ratio = improved / len(subset) if subset else 0.0
            verdict_improved_ratio.append(ratio)

            # Average feedback delta as proxy for mAP improvement over time
            feedback_deltas = []
            for r in subset:
                feedback = r.get("feedback", {})
                delta = feedback.get("delta", {})
                if "mAP" in delta:
                    feedback_deltas.append(delta["mAP"])
                elif "avg_confidence" in delta:
                    feedback_deltas.append(delta["avg_confidence"])

            avg_delta = float(np.mean(feedback_deltas)) if feedback_deltas else 0.0
            mAP_list.append(avg_delta)

        # Ensure last point is included
        if num_images_list and num_images_list[-1] != len(valid_results):
            num_images_list.append(len(valid_results))
            improved = sum(1 for r in valid_results if r.get("verdict") == "improved")
            verdict_improved_ratio.append(improved / len(valid_results))
            feedback_deltas = []
            for r in valid_results:
                feedback = r.get("feedback", {})
                delta = feedback.get("delta", {})
                if "mAP" in delta:
                    feedback_deltas.append(delta["mAP"])
                elif "avg_confidence" in delta:
                    feedback_deltas.append(delta["avg_confidence"])
            mAP_list.append(float(np.mean(feedback_deltas)) if feedback_deltas else 0.0)

        curve = {
            "num_images": num_images_list,
            "mAP_delta": mAP_list,
            "improved_ratio": verdict_improved_ratio,
        }

        logger.info(
            f"Learning curve: {len(num_images_list)} points, "
            f"final improved_ratio={verdict_improved_ratio[-1]:.3f}" if verdict_improved_ratio else ""
        )

        return curve

    @staticmethod
    def generate_report(evaluation_results: Dict, output_path: str):
        """Generate evaluation report (JSON format).

        Args:
            evaluation_results: Evaluation results dictionary.
            output_path: Output file path.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Convert numpy types for serialization
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        report = convert(evaluation_results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation report saved to {output_path}")

    @staticmethod
    def _load_annotations(annotation_file: str) -> Dict:
        """Load COCO format annotations."""
        if not os.path.exists(annotation_file):
            logger.warning(f"Annotation file not found: {annotation_file}")
            return {}

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        id_to_filename = {}
        for img_info in coco_data.get("images", []):
            id_to_filename[img_info["id"]] = img_info["file_name"]

        annotations = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            filename = id_to_filename.get(img_id)
            if filename is None:
                continue

            if filename not in annotations:
                annotations[filename] = []

            bbox = ann["bbox"]
            x1, y1, w, h = bbox
            annotations[filename].append({
                "bbox": [x1, y1, x1 + w, y1 + h],
                "class_id": ann.get("category_id", 0),
            })

        return annotations
