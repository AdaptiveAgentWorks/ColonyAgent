"""Skill building module - generates Skill Markdown documents from successful trajectories.

Based on XSkill's skill_builder.py, adapted for microbial image processing scenarios.
"""

import json
import os
import re
from typing import Dict, List, Optional

from loguru import logger


# --------- Prompt Templates ---------

GENERATE_SKILL_PROMPT = """\
You are an expert at creating reusable microscopy image processing skills.

Below are processing trajectories with their verdicts (improved/degraded) and quality patterns. \
Generate a Skill document in Markdown format that captures the reusable \
processing strategy, including both when to apply and when NOT to apply.

## Trajectories (verdict indicates if preprocessing helped or hurt detection)
{trajectories_text}

## Quality Patterns
{quality_patterns}

## Instructions
Create a Markdown skill document with the following structure:
1. **Skill Name**: A descriptive name for this processing strategy
2. **When to Apply**: Quality conditions under which this skill IS helpful (improved verdict)
3. **When NOT to Apply**: Quality conditions under which this skill HURTS detection (degraded verdict)
4. **Processing Steps**: Ordered list of tool calls with recommended parameters
5. **Expected Outcomes**: What quality improvements to expect when applied correctly
6. **Pitfalls**: Common mistakes to avoid, including conditions where preprocessing degrades results

Keep it concise and actionable. Focus on the causal relationship between \
image quality problems and effective tool sequences. Include both positive and negative examples.
"""

MERGE_SKILL_PROMPT = """\
You are an expert at consolidating microscopy image processing skills.

Below are two skill documents that have overlapping content. Merge them into \
a single, comprehensive skill document while removing redundancy.

## Skill 1
{skill_1}

## Skill 2
{skill_2}

## Instructions
Produce a single merged Markdown skill document that:
- Combines all unique processing strategies from both skills
- Removes duplicate content
- Maintains clear structure (When to Apply, Steps, Expected Outcomes, Pitfalls)
- Is concise and actionable
"""


class SkillBuilder:
    """Skill builder - generates reusable Skill documents from successful trajectories."""

    def __init__(self, llm_client, save_dir: str):
        """Initialize skill builder.

        Args:
            llm_client: LLMClient instance.
            save_dir: Skill document save directory.
        """
        self.llm = llm_client
        self.save_dir = save_dir
        self.skills: Dict[str, str] = {}  # skill_name -> markdown content

        os.makedirs(save_dir, exist_ok=True)

    def generate_skill(
        self,
        successful_trajectories: List[Dict],
        quality_patterns: Dict,
    ) -> str:
        """Generate Skill Markdown document from successful trajectories.

        Args:
            successful_trajectories: Successful trajectory list, each item contains:
                - trajectory: Trajectory step list
                - quality_before: Quality before processing
                - quality_after: Quality after processing
                - summary: Trajectory summary text
            quality_patterns: Quality patterns dict, e.g.:
                {
                    "common_problems": ["low blur", "high noise"],
                    "effective_tools": ["deconvolution", "denoise"],
                    "quality_improvements": {"blur": +0.3, "noise": +0.2}
                }

        Returns:
            Generated Skill Markdown text.
        """
        # Format trajectories
        traj_parts = []
        for i, traj in enumerate(successful_trajectories):
            summary = traj.get("summary", "No summary available")
            q_before = traj.get("quality_before", {})
            q_after = traj.get("quality_after", {})
            verdict = traj.get("verdict", "unknown")
            verdict_emoji = "✓" if verdict == "improved" else "✗" if verdict == "degraded" else "?"
            traj_parts.append(
                f"### Trajectory {i + 1} {verdict_emoji} (verdict: {verdict})\n"
                f"**Before**: {json.dumps(q_before, ensure_ascii=False)}\n"
                f"**After**: {json.dumps(q_after, ensure_ascii=False)}\n"
                f"**Summary**: {summary}\n"
            )
        trajectories_text = "\n".join(traj_parts) if traj_parts else "No trajectories provided."

        # Format quality patterns
        patterns_text = json.dumps(quality_patterns, indent=2, ensure_ascii=False)

        prompt = GENERATE_SKILL_PROMPT.format(
            trajectories_text=trajectories_text,
            quality_patterns=patterns_text,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            skill_content = self.llm.chat(messages, temperature=0.3)

            # Clean markdown wrapper
            skill_content = self._clean_markdown(skill_content)

            # Extract skill name
            skill_name = self._extract_skill_name(skill_content)
            self.skills[skill_name] = skill_content

            logger.info(f"Generated skill: {skill_name}")
            return skill_content
        except Exception as e:
            logger.error(f"Failed to generate skill: {e}")
            return ""

    def merge_skills(self, skill_1: str, skill_2: str) -> str:
        """Merge two similar Skills using LLM.

        Args:
            skill_1: Markdown content of the first Skill.
            skill_2: Markdown content of the second Skill.

        Returns:
            Merged Skill Markdown text.
        """
        prompt = MERGE_SKILL_PROMPT.format(skill_1=skill_1, skill_2=skill_2)
        messages = [{"role": "user", "content": prompt}]

        try:
            merged = self.llm.chat(messages, temperature=0.2)
            merged = self._clean_markdown(merged)

            # Update skill name
            skill_name = self._extract_skill_name(merged)
            self.skills[skill_name] = merged

            logger.info(f"Merged skills into: {skill_name}")
            return merged
        except Exception as e:
            logger.error(f"Failed to merge skills: {e}")
            return skill_1  # Return first one on failure

    def save(self):
        """Save all Skills to Markdown files."""
        for skill_name, content in self.skills.items():
            # Convert skill name to valid filename
            filename = re.sub(r"[^\w\s-]", "", skill_name).strip()
            filename = re.sub(r"[\s]+", "_", filename).lower()
            if not filename:
                filename = "unnamed_skill"
            filepath = os.path.join(self.save_dir, f"{filename}.md")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        # Also save a summary file
        all_skills_path = os.path.join(self.save_dir, "SKILLS.md")
        with open(all_skills_path, "w", encoding="utf-8") as f:
            f.write("# Microscopy Image Processing Skills\n\n")
            for skill_name, content in self.skills.items():
                f.write(f"---\n\n{content}\n\n")

        logger.info(f"Saved {len(self.skills)} skills to {self.save_dir}")

    def load(self):
        """Load all Skills from Markdown files."""
        if not os.path.exists(self.save_dir):
            logger.info(f"Skill directory {self.save_dir} not found")
            return

        loaded = 0
        for filename in os.listdir(self.save_dir):
            if filename.endswith(".md") and filename != "SKILLS.md":
                filepath = os.path.join(self.save_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                skill_name = self._extract_skill_name(content)
                self.skills[skill_name] = content
                loaded += 1

        logger.info(f"Loaded {loaded} skills from {self.save_dir}")

    def get_all_skills(self) -> str:
        """Return complete Skill document.

        Returns:
            Merged Markdown text of all Skills.
        """
        if not self.skills:
            return ""

        parts = ["# Microscopy Image Processing Skills\n"]
        for skill_name, content in self.skills.items():
            parts.append(f"---\n\n{content}\n")

        return "\n".join(parts)

    def _clean_markdown(self, text: str) -> str:
        """Clean markdown wrapper returned by LLM.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        text = re.sub(r"^```markdown\n", "", text)
        text = re.sub(r"^```\n", "", text)
        text = re.sub(r"\n```$", "", text)
        return text.strip()

    def _extract_skill_name(self, content: str) -> str:
        """Extract skill name from Skill document.

        Args:
            content: Markdown content

        Returns:
            Skill name
        """
        # Try extracting from H1 heading
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Try extracting from H2 heading
        match = re.search(r"^##\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Use first 50 characters
        first_line = content.split("\n")[0].strip()
        return first_line[:50] if first_line else "Unnamed Skill"
