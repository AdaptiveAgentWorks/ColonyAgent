"""Closed-loop feedback module.

Evaluates preprocessing effectiveness (baseline vs. enhanced), generates experience update operations,
and implements automatic iterative optimization of experiences.
"""

import json
import re
import time
from typing import List, Dict, Optional, Any

import numpy as np
from loguru import logger

from tools.registry import get_tool, list_tools
from tools.base import ToolResult
from utils.metrics import (
    calculate_map,
    calculate_recall,
    calculate_f1,
    compare_detections,
)
from prompts.feedback_prompts import FEEDBACK_ANALYSIS_PROMPT
from agent.colony_agent import AgentResult


class FeedbackLoop:
    """Closed-loop feedback - evaluates preprocessing effectiveness and generates experience updates."""

    # Proxy metric thresholds
    IMPROVEMENT_THRESHOLD = 0.005   # Improvement delta above this value is considered improved
    DEGRADATION_THRESHOLD = -0.005  # Improvement delta below this value is considered degraded

    def __init__(self, llm_client, quality_assessor, config: dict = None):
        """
        Args:
            llm_client: LLMClient instance.
            quality_assessor: ImageQualityAssessor instance.
            config: Optional configuration.
        """
        self.llm = llm_client
        self.assessor = quality_assessor
        self.config = config or {}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, agent_result: AgentResult, ground_truth: List[Dict] = None
    ) -> dict:
        """Evaluate preprocessing effectiveness.

        1. Detect on original image with the same detector -> baseline
        2. agent_result already has post-processing detection results -> enhanced
        3. When ground_truth is available, calculate mAP/Recall comparison
        4. When ground_truth is unavailable, use proxy metrics

        Args:
            agent_result: Agent processing result.
            ground_truth: Optional ground truth annotation list, each item contains bbox, class_id.

        Returns:
            FeedbackResult dict.
        """
        # Baseline detection
        baseline_dets = self._detect_on_original(
            agent_result.original_image, agent_result.detector_used
        )
        enhanced_dets = agent_result.detections

        if ground_truth:
            # With ground truth: full metrics comparison
            comparison = compare_detections(baseline_dets, enhanced_dets, ground_truth)

            delta = comparison["improvement"]
            # Comprehensive judgment
            f1_delta = delta.get("f1_delta", 0.0)
            if f1_delta > self.IMPROVEMENT_THRESHOLD:
                verdict = "improved"
            elif f1_delta < self.DEGRADATION_THRESHOLD:
                verdict = "degraded"
            else:
                verdict = "neutral"

            return {
                "verdict": verdict,
                "baseline_metrics": comparison["baseline"],
                "enhanced_metrics": comparison["enhanced"],
                "delta": {
                    "mAP": delta["mAP_delta"],
                    "recall": delta["recall_delta"],
                    "f1": delta["f1_delta"],
                    "detection_count": delta["detection_count_delta"],
                },
                "has_ground_truth": True,
                "details": (
                    f"F1 delta={f1_delta:+.4f}, "
                    f"mAP delta={delta['mAP_delta']:+.4f}, "
                    f"recall delta={delta['recall_delta']:+.4f}"
                ),
            }
        else:
            # Without ground truth: proxy metrics
            proxy = self._compute_proxy_metrics(baseline_dets, enhanced_dets)

            # Comprehensive judgment (based on confidence and detection count changes)
            conf_delta = proxy.get("avg_confidence_delta", 0.0)
            high_conf_delta = proxy.get("high_confidence_ratio_delta", 0.0)
            count_delta = proxy.get("count_delta", 0.0)
            baseline_count = proxy.get("baseline_count", 1)

            # Normalize detection count change: avoid large count differences dominating the score
            # If detection count drops more than 30%, consider it degraded even if confidence improved
            count_change_ratio = count_delta / max(baseline_count, 1)
            if count_change_ratio < -0.3:
                # Significant count drop, clearly degraded
                score = -0.5
            elif count_change_ratio > 0.3:
                # Significant count increase, judge quality by confidence
                score = conf_delta * 0.6 + high_conf_delta * 0.4
            else:
                # Count change is minor, judge quality by confidence
                score = conf_delta * 0.6 + high_conf_delta * 0.4

            if score > self.IMPROVEMENT_THRESHOLD:
                verdict = "improved"
            elif score < self.DEGRADATION_THRESHOLD:
                verdict = "degraded"
            else:
                verdict = "neutral"

            return {
                "verdict": verdict,
                "baseline_metrics": {
                    "num_detections": proxy["baseline_count"],
                    "avg_confidence": proxy["baseline_avg_confidence"],
                    "high_confidence_ratio": proxy["baseline_high_confidence_ratio"],
                },
                "enhanced_metrics": {
                    "num_detections": proxy["enhanced_count"],
                    "avg_confidence": proxy["enhanced_avg_confidence"],
                    "high_confidence_ratio": proxy["enhanced_high_confidence_ratio"],
                },
                "delta": {
                    "avg_confidence": proxy["avg_confidence_delta"],
                    "detection_count": proxy["count_delta"],
                    "high_confidence_ratio": proxy["high_confidence_ratio_delta"],
                },
                "has_ground_truth": False,
                "details": (
                    f"Proxy score={score:+.4f}, "
                    f"confidence delta={conf_delta:+.4f}, "
                    f"count delta={proxy['count_delta']:+d}"
                ),
            }

    # ------------------------------------------------------------------
    # Experience update generation
    # ------------------------------------------------------------------

    def generate_experience_updates(
        self, agent_result: AgentResult, feedback_result: dict
    ) -> List[Dict]:
        """Transform feedback into experience update operations.

        1. Use LLM to analyze trajectory + feedback results
        2. Generate experience update operations [{"action": "add"/"modify"/..., "experience": {...}}]

        If LLM is unavailable, generate using simple rules.

        Args:
            agent_result: Agent processing result.
            feedback_result: Return value from evaluate().

        Returns:
            Experience update operation list.
        """
        try:
            return self._generate_updates_with_llm(agent_result, feedback_result)
        except Exception as e:
            logger.warning(f"LLM experience update generation failed ({e}), using rules")
            return self._generate_updates_with_rules(agent_result, feedback_result)

    def _generate_updates_with_llm(
        self, agent_result: AgentResult, feedback_result: dict
    ) -> List[Dict]:
        """Analyze and generate experience updates using LLM."""
        # Format trajectory
        trajectory_text = json.dumps(agent_result.trajectory, indent=2, ensure_ascii=False, default=str)

        # Format metrics
        baseline_text = json.dumps(
            feedback_result.get("baseline_metrics", {}), indent=2, ensure_ascii=False
        )
        enhanced_text = json.dumps(
            feedback_result.get("enhanced_metrics", {}), indent=2, ensure_ascii=False
        )
        quality_text = json.dumps(
            agent_result.quality_report, indent=2, ensure_ascii=False
        )

        prompt = FEEDBACK_ANALYSIS_PROMPT.format(
            trajectory=trajectory_text,
            baseline_metrics=baseline_text,
            enhanced_metrics=enhanced_text,
            quality_report=quality_text,
        )

        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm.chat(messages, temperature=0.3)

        # Extract JSON from response
        updates = self._extract_experience_updates(response_text)
        logger.info(f"LLM generated {len(updates)} experience updates")
        return updates

    def _extract_experience_updates(self, text: str) -> List[Dict]:
        """Extract experience update operations from LLM response text."""
        # Try to extract JSON blocks
        json_blocks = re.findall(r"```json\s*(.*?)```", text, re.DOTALL)
        if not json_blocks:
            # Try matching curly braces
            json_blocks = re.findall(r"\{[\s\S]*\"experience_updates\"[\s\S]*?\}", text)

        for block in json_blocks:
            block = block.strip()
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue

            raw_updates = data.get("experience_updates", [])
            if not raw_updates:
                continue

            # Standardize format
            updates = []
            for item in raw_updates:
                op = item.get("operation", "add")
                update = {"action": op}

                if op in ("add", "modify"):
                    update["experience"] = {
                        "condition": item.get("condition", ""),
                        "action": item.get("action", ""),
                        "reason": item.get("reason", ""),
                    }
                    if "experience_id" in item:
                        update["experience_id"] = item["experience_id"]
                elif op in ("reinforce", "weaken"):
                    update["experience_id"] = item.get("experience_id", "")
                    update["reason"] = item.get("reason", "")

                updates.append(update)

            return updates

        return []

    def _generate_updates_with_rules(
        self, agent_result: AgentResult, feedback_result: dict
    ) -> List[Dict]:
        """Generate experience updates based on simple rules (fallback when LLM is unavailable)."""
        updates = []
        verdict = feedback_result.get("verdict", "neutral")
        quality = agent_result.quality_report
        tools_used = [
            step["tool"]
            for step in agent_result.trajectory
            if step.get("type") == "preprocessing" and step.get("success", False)
        ]

        if not tools_used:
            return updates

        # Build quality condition description
        low_dims = []
        dim_map = {
            "blur_score": "blur",
            "brightness_score": "brightness",
            "contrast_score": "contrast",
            "noise_score": "noise",
            "color_bias_score": "color_bias",
        }
        for key, label in dim_map.items():
            val = quality.get(key, 1.0)
            if val < 0.5:
                low_dims.append(f"{label} < {val:.2f}")

        condition = f"When {', '.join(low_dims)}" if low_dims else "General condition"
        action_desc = f"Apply: {' -> '.join(tools_used)}"

        if verdict == "improved":
            updates.append({
                "action": "add",
                "experience": {
                    "condition": condition,
                    "action": action_desc,
                    "reason": (
                        f"This tool sequence improved detection "
                        f"(verdict={verdict}). "
                        f"Delta: {json.dumps(feedback_result.get('delta', {}))}"
                    ),
                },
            })
        elif verdict == "degraded":
            updates.append({
                "action": "add",
                "experience": {
                    "condition": condition,
                    "action": f"AVOID: {action_desc}",
                    "reason": (
                        f"This tool sequence degraded detection "
                        f"(verdict={verdict}). "
                        f"Delta: {json.dumps(feedback_result.get('delta', {}))}"
                    ),
                },
            })

        return updates

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _compute_proxy_metrics(
        self, baseline_dets: List[Dict], enhanced_dets: List[Dict]
    ) -> dict:
        """Proxy metrics when ground truth is unavailable.

        Calculates:
        - Average confidence change
        - Detection count change
        - High-confidence detection ratio change (confidence > 0.5)
        """
        HIGH_CONF_THRESHOLD = 0.5

        # Baseline
        baseline_count = len(baseline_dets)
        baseline_confs = [d.get("confidence", 0.0) for d in baseline_dets]
        baseline_avg_conf = float(np.mean(baseline_confs)) if baseline_confs else 0.0
        baseline_high = sum(1 for c in baseline_confs if c > HIGH_CONF_THRESHOLD)
        baseline_high_ratio = baseline_high / baseline_count if baseline_count > 0 else 0.0

        # Enhanced
        enhanced_count = len(enhanced_dets)
        enhanced_confs = [d.get("confidence", 0.0) for d in enhanced_dets]
        enhanced_avg_conf = float(np.mean(enhanced_confs)) if enhanced_confs else 0.0
        enhanced_high = sum(1 for c in enhanced_confs if c > HIGH_CONF_THRESHOLD)
        enhanced_high_ratio = enhanced_high / enhanced_count if enhanced_count > 0 else 0.0

        return {
            "baseline_count": baseline_count,
            "baseline_avg_confidence": baseline_avg_conf,
            "baseline_high_confidence_ratio": baseline_high_ratio,
            "enhanced_count": enhanced_count,
            "enhanced_avg_confidence": enhanced_avg_conf,
            "enhanced_high_confidence_ratio": enhanced_high_ratio,
            "count_delta": enhanced_count - baseline_count,
            "avg_confidence_delta": enhanced_avg_conf - baseline_avg_conf,
            "high_confidence_ratio_delta": enhanced_high_ratio - baseline_high_ratio,
        }

    def _detect_on_original(
        self, original_image: np.ndarray, detector_name: str
    ) -> List[Dict]:
        """Detect on original image using specified detector (as baseline).

        Args:
            original_image: Original image.
            detector_name: Detector tool name.

        Returns:
            Detection results list.
        """
        # Read detector parameters from config
        detectors_config = self.config.get("detectors", {})
        detector_defaults = {
            "yolov8_detect": detectors_config.get("yolov8", {}),
            "rtdetr_detect": detectors_config.get("rt_detr", {}),
            "fasterrcnn_detect": detectors_config.get("faster_rcnn", {}),
        }
        try:
            tool = get_tool(detector_name)
            params = detector_defaults.get(detector_name, {})
            result: ToolResult = tool.call(original_image, **params)
            if result.success:
                return result.detections
            else:
                logger.warning(
                    f"Baseline detection with {detector_name} failed: {result.error}"
                )
                return []
        except KeyError:
            logger.error(f"Detector '{detector_name}' not registered for baseline detection")
            return []
        except Exception as e:
            logger.error(f"Baseline detection failed: {e}")
            return []
