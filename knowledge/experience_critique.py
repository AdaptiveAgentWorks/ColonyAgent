"""Cross-rollout critique module - compares multiple rollout summaries and extracts causal experiences.

Based on XSkill's experience_critique.py, adapted for microbial image processing scenarios.
"""

import json
import re
import uuid
from typing import List, Dict, Optional

from loguru import logger


# --------- Prompt Templates ---------

CRITIQUE_PROMPT = """\
You are an expert at analyzing microscopy image processing results and extracting reusable experiences.

Below are summaries from multiple processing rollouts for the **same** microscopy image. \
Some rollouts may have succeeded and some may have failed. By comparing them, extract \
causal experiences that can guide future image processing.

## Ground Truth
{ground_truth}

## Rollout Summaries
{rollout_summaries}

{existing_experiences_section}

## Instructions
Compare the successful and failed rollouts. For each insight, determine:
1. **condition**: Under what image quality conditions does this apply? (e.g., "low blur_score (<0.3)")
2. **action**: What specific processing action should be taken? (e.g., "apply deconvolution with kernel_size=5")
3. **reason**: Why does this work? What causal relationship was observed?

Output a JSON list of experience update operations:
```json
[
    {{
        "action": "add",
        "experience": {{
            "condition": "...",
            "action": "...",
            "reason": "...",
            "source": "critique"
        }}
    }},
    {{
        "action": "modify",
        "exp_id": "existing_exp_id",
        "experience": {{
            "condition": "...",
            "action": "...",
            "reason": "...",
            "source": "critique"
        }}
    }}
]
```

Rules:
- Use "add" for new experiences, "modify" to update an existing one (only if closely related).
- Be specific about conditions (reference quality metrics) and actions (reference tool names and parameters).
- Only extract genuinely causal relationships, not coincidences.
- Limit to at most 3 new operations.
"""


class ExperienceCritic:
    """Experience critic - compares successful/failed trajectories and extracts causal experiences."""

    def __init__(self, llm_client):
        """Initialize critic.

        Args:
            llm_client: LLMClient instance.
        """
        self.llm = llm_client

    def critique(
        self,
        rollout_summaries: List[str],
        ground_truth: dict,
        existing_experiences: List[Dict] = None,
    ) -> List[Dict]:
        """Compare multiple rollout summaries and extract experience update operations.

        Args:
            rollout_summaries: List of multiple rollout summary texts for the same image.
            ground_truth: Ground truth annotation.
            existing_experiences: List of existing experiences (optional), used to determine add or modify.

        Returns:
            Experience update operation list, each item format:
            {
                "action": "add" | "modify",
                "exp_id": "..." (for modify),
                "experience": {
                    "id": str,
                    "condition": str,
                    "action": str,
                    "reason": str,
                    "source": "critique"
                }
            }
        """
        if not rollout_summaries:
            return []

        # Format rollout summaries
        formatted_summaries = []
        for i, summary in enumerate(rollout_summaries):
            formatted_summaries.append(f"### Rollout {i + 1}\n{summary}")
        summaries_text = "\n\n".join(formatted_summaries)

        # Format ground truth
        gt_text = json.dumps(ground_truth, indent=2, ensure_ascii=False) if ground_truth else "[Not available]"

        # Format existing experiences
        if existing_experiences:
            exp_lines = []
            for exp in existing_experiences:
                exp_lines.append(
                    f"- [{exp.get('id', '?')}] condition: {exp.get('condition', '?')}, "
                    f"action: {exp.get('action', '?')}, reason: {exp.get('reason', '?')}"
                )
            existing_section = (
                "## Existing Experiences\n"
                "Consider modifying these if your insight updates an existing one:\n"
                + "\n".join(exp_lines)
            )
        else:
            existing_section = ""

        prompt = CRITIQUE_PROMPT.format(
            ground_truth=gt_text,
            rollout_summaries=summaries_text,
            existing_experiences_section=existing_section,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm.chat(messages, temperature=0.3)
            operations = self._parse_response(response)
            logger.info(f"Critique generated {len(operations)} experience operations")
            return operations
        except Exception as e:
            logger.error(f"Failed to generate critique: {e}")
            return []

    def _parse_response(self, response: str) -> List[Dict]:
        """Parse JSON operation list returned by LLM.

        Args:
            response: LLM's raw response text.

        Returns:
            Parsed operation list.
        """
        # Try to extract JSON from markdown code block
        try:
            if "```json" in response:
                payload = response.split("```json")[-1].split("```")[0].strip()
            elif "```" in response:
                payload = response.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON array directly
                match = re.search(r"\[.*\]", response, re.DOTALL)
                payload = match.group(0) if match else "[]"

            ops = json.loads(payload)
            if not isinstance(ops, list):
                logger.warning("Critique response is not a list, wrapping")
                ops = [ops] if isinstance(ops, dict) else []
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse critique response: {e}")
            return []

        # Validate and complete each operation
        validated = []
        for op in ops:
            action = op.get("action", "add")
            if action not in ("add", "modify"):
                continue

            experience = op.get("experience", {})
            if not experience.get("condition") or not experience.get("action"):
                continue

            # Ensure id exists
            if "id" not in experience:
                experience["id"] = f"exp_{uuid.uuid4().hex[:8]}"

            # Ensure source exists
            if "source" not in experience:
                experience["source"] = "critique"

            validated_op = {"action": action, "experience": experience}
            if action == "modify" and "exp_id" in op:
                validated_op["exp_id"] = op["exp_id"]

            validated.append(validated_op)

        return validated
