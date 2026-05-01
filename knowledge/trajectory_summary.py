"""Trajectory summary module - summarizes a complete processing trajectory into structured text.

Based on XSkill's trajectory_summary.py, adapted for microbial image processing scenarios.
"""

import json
from typing import List, Dict, Optional

from loguru import logger


# --------- Prompt Templates ---------

TRAJECTORY_SUMMARY_PROMPT = """\
You are an expert at analyzing microscopy image processing trajectories.

Below is a complete processing trajectory for a microscopy image, including tool calls, \
quality assessments, and feedback. Summarize this trajectory into a structured report.

## Trajectory
{trajectory_text}

## Quality Report (after processing)
{quality_report}

## Detection Result
{detection_result}

{ground_truth_section}

## Instructions
Please provide a structured summary covering:
1. **Initial State**: What was the initial image quality? What problems were identified?
2. **Processing Steps**: List each tool call, its parameters, and outcome (improved/degraded/neutral).
3. **Key Decisions**: What were the critical decision points? Why was each tool chosen?
4. **Final Outcome**: Was the processing successful? How did detection results compare to ground truth (if available)?
5. **Lessons Learned**: What worked well? What could be improved?

Format your response as a structured text summary (not JSON).
"""


class TrajectorySummarizer:
    """Trajectory summarizer - uses LLM to summarize processing trajectories into structured text."""

    def __init__(self, llm_client):
        """Initialize summarizer.

        Args:
            llm_client: LLMClient instance for calling LLM.
        """
        self.llm = llm_client

    def summarize(
        self,
        trajectory: List[Dict],
        quality_report: dict,
        detection_result: dict,
        ground_truth: dict = None,
    ) -> str:
        """Summarize a complete processing trajectory.

        Args:
            trajectory: Processing trajectory list, each dict contains:
                - step: Step number
                - type: Type (quality_assess / tool_call / feedback)
                - tool: Tool name (for tool_call type)
                - params: Tool parameters (for tool_call type)
                - result: Execution result
            quality_report: Final quality assessment report (QualityReport in dict format)
            detection_result: Detection result
            ground_truth: Ground truth annotation (optional)

        Returns:
            Structured text summary
        """
        # Format trajectory
        trajectory_text = self._format_trajectory(trajectory)

        # Format quality report
        quality_text = json.dumps(quality_report, indent=2, ensure_ascii=False)

        # Format detection result
        detection_text = json.dumps(detection_result, indent=2, ensure_ascii=False)

        # Ground truth section
        if ground_truth:
            gt_text = f"## Ground Truth\n{json.dumps(ground_truth, indent=2, ensure_ascii=False)}"
        else:
            gt_text = ""

        prompt = TRAJECTORY_SUMMARY_PROMPT.format(
            trajectory_text=trajectory_text,
            quality_report=quality_text,
            detection_result=detection_text,
            ground_truth_section=gt_text,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            summary = self.llm.chat(messages, temperature=0.3)
            logger.info(f"Trajectory summary generated ({len(summary)} chars)")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate trajectory summary: {e}")
            # Fallback: generate simple text summary
            return self._fallback_summary(trajectory, quality_report, detection_result)

    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """Format trajectory into readable text.

        Args:
            trajectory: Trajectory step list

        Returns:
            Formatted text
        """
        lines = []
        for step_data in trajectory:
            step = step_data.get("step", "?")
            step_type = step_data.get("type", "unknown")

            if step_type == "quality_assess":
                result = step_data.get("result", {})
                lines.append(
                    f"Step {step} [Quality Assessment]: "
                    f"overall={result.get('overall_score', 'N/A'):.3f}, "
                    f"blur={result.get('blur_score', 'N/A'):.3f}, "
                    f"brightness={result.get('brightness_score', 'N/A'):.3f}, "
                    f"contrast={result.get('contrast_score', 'N/A'):.3f}, "
                    f"noise={result.get('noise_score', 'N/A'):.3f}, "
                    f"color_bias={result.get('color_bias_score', 'N/A'):.3f}"
                )
            elif step_type == "tool_call":
                tool = step_data.get("tool", "unknown")
                params = step_data.get("params", {})
                result = step_data.get("result", {})
                success = result.get("success", "N/A")
                metadata = result.get("metadata", {})
                lines.append(
                    f"Step {step} [Tool Call]: {tool}({json.dumps(params, ensure_ascii=False)}) "
                    f"-> success={success}, metadata={json.dumps(metadata, ensure_ascii=False)}"
                )
            elif step_type == "feedback":
                result = step_data.get("result", "")
                lines.append(f"Step {step} [Feedback]: {result}")
            else:
                lines.append(
                    f"Step {step} [{step_type}]: {json.dumps(step_data.get('result', ''), ensure_ascii=False)}"
                )

        return "\n".join(lines)

    def _fallback_summary(
        self,
        trajectory: List[Dict],
        quality_report: dict,
        detection_result: dict,
    ) -> str:
        """Generate simple fallback summary when LLM call fails.

        Args:
            trajectory: Trajectory step list
            quality_report: Quality report
            detection_result: Detection result

        Returns:
            Simple text summary
        """
        tool_calls = [s for s in trajectory if s.get("type") == "tool_call"]
        tools_used = [s.get("tool", "unknown") for s in tool_calls]
        overall = quality_report.get("overall_score", "N/A")
        num_detections = len(detection_result.get("detections", []))

        return (
            f"Trajectory Summary (fallback):\n"
            f"- Total steps: {len(trajectory)}\n"
            f"- Tools used: {', '.join(tools_used) if tools_used else 'None'}\n"
            f"- Final quality score: {overall}\n"
            f"- Detections found: {num_detections}\n"
        )
