"""Skill adaptation module - adapts Skill templates to current scenario (for Phase II use).

Based on current image quality and retrieved experiences, adapts base Skill documents,
and generates customized processing guidance to inject into Agent's system prompt.
"""

import json
from typing import Dict, List

from loguru import logger


# --------- Prompt Templates ---------

ADAPT_SKILL_PROMPT = """\
You are an expert at adapting microscopy image processing skills to specific situations.

Below is a base skill document, the current image's quality report, and \
retrieved experiences from similar past cases. Adapt the base skill to \
create a customized processing guide for this specific image.

## Base Skill Document
{base_skill}

## Current Image Quality Report
{quality_report}

## Retrieved Experiences
{experiences_text}

## Instructions
Adapt the base skill by:
1. **Prioritize** steps that address the current image's specific quality issues
2. **Adjust parameters** based on the severity of quality problems
3. **Incorporate lessons** from retrieved experiences
4. **Remove irrelevant** steps that don't apply to this image's conditions
5. **Add warnings** from past failures in similar conditions

Output the adapted skill as a concise Markdown document suitable for \
injection into an agent's system prompt. Keep it actionable and specific \
to this image's needs.
"""


class SkillAdapter:
    """Skill adapter - adapts general Skills to current scenario."""

    def __init__(self, llm_client):
        """Initialize adapter.

        Args:
            llm_client: LLMClient instance.
        """
        self.llm = llm_client

    def adapt(
        self,
        base_skill: str,
        quality_report: dict,
        retrieved_experiences: List[Dict],
    ) -> str:
        """Adapt Skill template based on current image quality and retrieved experiences.

        Args:
            base_skill: Base Skill Markdown document.
            quality_report: Current image's quality assessment report (dict representation of QualityReport).
            retrieved_experiences: List of retrieved related experiences.

        Returns:
            Adapted Skill text, can be injected into Agent system prompt.
        """
        if not base_skill:
            logger.warning("No base skill provided, returning empty adaptation")
            return ""

        # Format quality report
        quality_text = json.dumps(quality_report, indent=2, ensure_ascii=False)

        # Format experiences
        if retrieved_experiences:
            exp_parts = []
            for i, exp in enumerate(retrieved_experiences):
                sim = exp.get("_similarity", "N/A")
                exp_parts.append(
                    f"### Experience {i + 1} (similarity: {sim})\n"
                    f"- **Condition**: {exp.get('condition', 'N/A')}\n"
                    f"- **Action**: {exp.get('action', 'N/A')}\n"
                    f"- **Reason**: {exp.get('reason', 'N/A')}\n"
                    f"- **Success Rate**: {exp.get('success_rate', 'N/A')}"
                )
            experiences_text = "\n\n".join(exp_parts)
        else:
            experiences_text = "No relevant experiences found."

        prompt = ADAPT_SKILL_PROMPT.format(
            base_skill=base_skill,
            quality_report=quality_text,
            experiences_text=experiences_text,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            adapted = self.llm.chat(messages, temperature=0.3)

            # Clean markdown wrapper
            import re

            adapted = re.sub(r"^```markdown\n", "", adapted)
            adapted = re.sub(r"^```\n", "", adapted)
            adapted = re.sub(r"\n```$", "", adapted)
            adapted = adapted.strip()

            logger.info(f"Adapted skill generated ({len(adapted)} chars)")
            return adapted
        except Exception as e:
            logger.error(f"Failed to adapt skill: {e}")
            return base_skill  # Return original skill on failure
