"""Detection evaluation metrics calculation.

predictions format: List[Dict], each dict contains bbox (xyxy), confidence, class_id
ground_truths format: List[Dict], each dict contains bbox (xyxy), class_id
"""

import numpy as np
from typing import List, Dict, Optional


def calculate_iou(box1: list, box2: list) -> float:
    """Calculate IoU (Intersection over Union) of two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format

    Returns:
        IoU value (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """Calculate Average Precision (AP) for a single class.

    Uses all-point interpolation method.

    Args:
        predictions: Prediction list, each item contains bbox, confidence, class_id
        ground_truths: Ground truth list, each item contains bbox, class_id
        iou_threshold: IoU matching threshold

    Returns:
        AP value (0-1)
    """
    if len(ground_truths) == 0:
        return 0.0 if len(predictions) > 0 else 1.0
    if len(predictions) == 0:
        return 0.0

    # Sort by confidence in descending order
    preds_sorted = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))
    matched_gt = set()

    for i, pred in enumerate(preds_sorted):
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truths):
            if j in matched_gt:
                continue
            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1

    # Cumulative
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # All-point interpolation AP
    # Add 0 before recall, 1 before precision
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Make precision monotonically decreasing (take max from right to left)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Calculate area at points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return float(ap)


def calculate_precision_recall_f1(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict:
    """Calculate precision, recall, f1 (computed per class then averaged).

    Args:
        predictions: Prediction list, each item contains bbox (xyxy), confidence, class_id
        ground_truths: Ground truth list, each item contains bbox (xyxy), class_id
        iou_threshold: IoU matching threshold

    Returns:
        {precision, recall, f1, tp, fp, fn, per_class}
    """
    if len(ground_truths) == 0:
        if len(predictions) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predictions), "fn": 0, "per_class": {}}

    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ground_truths), "per_class": {}}

    # Calculate per class separately, then average
    all_classes = set(g["class_id"] for g in ground_truths) | set(p["class_id"] for p in predictions)

    per_class_results = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for cls_id in sorted(all_classes):
        cls_preds = [p for p in predictions if p["class_id"] == cls_id]
        cls_gts = [g for g in ground_truths if g["class_id"] == cls_id]

        if len(cls_gts) == 0:
            # No GT for this class, all predictions are FP
            total_fp += len(cls_preds)
            continue

        if len(cls_preds) == 0:
            # No predictions for this class, all are FN
            total_fn += len(cls_gts)
            continue

        # Sort by confidence in descending order
        preds_sorted = sorted(cls_preds, key=lambda x: x.get("confidence", 1.0), reverse=True)
        matched_gt = set()
        tp, fp = 0, 0

        for pred in preds_sorted:
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt in enumerate(cls_gts):
                if j in matched_gt:
                    continue
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(cls_gts) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_cls_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_cls_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_results.append((cls_id, per_cls_prec, per_cls_rec))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": {cls_id: {"precision": p, "recall": r} for cls_id, p, r in per_class_results},
    }


def calculate_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: Optional[List[float]] = None,
) -> dict:
    """Calculate mAP (mean Average Precision).

    Computes AP per class then takes mean. Supports multiple IoU thresholds.

    Args:
        predictions: Prediction list
        ground_truths: Ground truth list
        iou_thresholds: IoU threshold list, default [0.5]

    Returns:
        dict containing:
            - mAP: Mean across all classes and all thresholds
            - per_class: AP for each class at each threshold
            - per_threshold: mAP for each threshold
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    # Collect all classes
    all_classes = set()
    for gt in ground_truths:
        all_classes.add(gt["class_id"])
    for pred in predictions:
        all_classes.add(pred["class_id"])

    if not all_classes:
        return {"mAP": 0.0, "per_class": {}, "per_threshold": {}}

    per_class = {}
    per_threshold = {t: [] for t in iou_thresholds}

    for cls_id in sorted(all_classes):
        cls_preds = [p for p in predictions if p["class_id"] == cls_id]
        cls_gts = [g for g in ground_truths if g["class_id"] == cls_id]

        per_class[cls_id] = {}
        for threshold in iou_thresholds:
            ap = calculate_ap(cls_preds, cls_gts, threshold)
            per_class[cls_id][threshold] = ap
            per_threshold[threshold].append(ap)

    # Calculate mAP for each threshold
    threshold_maps = {}
    for t in iou_thresholds:
        vals = per_threshold[t]
        threshold_maps[t] = float(np.mean(vals)) if vals else 0.0

    overall_map = float(np.mean(list(threshold_maps.values()))) if threshold_maps else 0.0

    return {
        "mAP": overall_map,
        "per_class": per_class,
        "per_threshold": threshold_maps,
    }


def calculate_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """Calculate recall.

    Args:
        predictions: Prediction list
        ground_truths: Ground truth list
        iou_threshold: IoU matching threshold

    Returns:
        Recall (0-1)
    """
    if len(ground_truths) == 0:
        return 1.0 if len(predictions) == 0 else 0.0
    if len(predictions) == 0:
        return 0.0

    matched_gt = set()

    # Sort by confidence in descending order
    preds_sorted = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    for pred in preds_sorted:
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truths):
            if j in matched_gt:
                continue
            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)

    return len(matched_gt) / len(ground_truths)


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score.

    Args:
        precision: Precision
        recall: Recall

    Returns:
        F1 score (0-1)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compare_detections(
    baseline_dets: List[Dict],
    enhanced_dets: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Compare two sets of detection results (baseline vs enhanced).

    Args:
        baseline_dets: Baseline detection results
        enhanced_dets: Enhanced detection results
        ground_truths: Ground truth annotations
        iou_threshold: IoU matching threshold

    Returns:
        dict containing metrics for both sets and their differences
    """
    # Baseline metrics
    baseline_map_result = calculate_map(baseline_dets, ground_truths, [iou_threshold])
    baseline_recall = calculate_recall(baseline_dets, ground_truths, iou_threshold)
    baseline_ap = baseline_map_result["mAP"]
    baseline_f1 = calculate_f1(baseline_ap, baseline_recall)

    # Enhanced metrics
    enhanced_map_result = calculate_map(enhanced_dets, ground_truths, [iou_threshold])
    enhanced_recall = calculate_recall(enhanced_dets, ground_truths, iou_threshold)
    enhanced_ap = enhanced_map_result["mAP"]
    enhanced_f1 = calculate_f1(enhanced_ap, enhanced_recall)

    return {
        "baseline": {
            "mAP": baseline_ap,
            "recall": baseline_recall,
            "f1": baseline_f1,
            "num_detections": len(baseline_dets),
            "detail": baseline_map_result,
        },
        "enhanced": {
            "mAP": enhanced_ap,
            "recall": enhanced_recall,
            "f1": enhanced_f1,
            "num_detections": len(enhanced_dets),
            "detail": enhanced_map_result,
        },
        "improvement": {
            "mAP_delta": enhanced_ap - baseline_ap,
            "recall_delta": enhanced_recall - baseline_recall,
            "f1_delta": enhanced_f1 - baseline_f1,
            "detection_count_delta": len(enhanced_dets) - len(baseline_dets),
        },
    }


# ===================================================================
# Colony Count Metrics
# ===================================================================


def calculate_colony_count_mae(pred_count: int, gt_count: int) -> float:
    """Colony count mean absolute error."""
    return abs(pred_count - gt_count)


def calculate_colony_count_rmse(pred_counts: List[int], gt_counts: List[int]) -> float:
    """Colony count root mean square error."""
    if len(pred_counts) != len(gt_counts):
        raise ValueError(
            f"pred_counts and gt_counts must have the same length, "
            f"got {len(pred_counts)} and {len(gt_counts)}"
        )
    if len(pred_counts) == 0:
        return 0.0
    squared_errors = [(p - g) ** 2 for p, g in zip(pred_counts, gt_counts)]
    return float(np.sqrt(np.mean(squared_errors)))


def calculate_recovery_rate(pred_count: int, gt_count: int) -> float:
    """Count recovery rate = pred / gt * 100%, returns 100% when gt=0 (if pred is also 0)."""
    if gt_count == 0:
        return 100.0 if pred_count == 0 else 0.0
    return pred_count / gt_count * 100.0


def batch_colony_metrics(results: List[Dict]) -> Dict:
    """
    Batch calculate colony count metrics.

    Args:
        results: List[Dict], each dict contains pred_count and gt_count

    Returns:
        {mae, rmse, avg_recovery_rate}
    """
    if not results:
        return {"mae": 0.0, "rmse": 0.0, "avg_recovery_rate": 0.0}

    pred_counts = [r["pred_count"] for r in results]
    gt_counts = [r["gt_count"] for r in results]

    # MAE
    maes = [calculate_colony_count_mae(p, g) for p, g in zip(pred_counts, gt_counts)]
    mae = float(np.mean(maes))

    # RMSE
    rmse = calculate_colony_count_rmse(pred_counts, gt_counts)

    # Average Recovery Rate
    recovery_rates = [
        calculate_recovery_rate(p, g) for p, g in zip(pred_counts, gt_counts)
    ]
    avg_recovery_rate = float(np.mean(recovery_rates))

    return {
        "mae": mae,
        "rmse": rmse,
        "avg_recovery_rate": avg_recovery_rate,
    }
