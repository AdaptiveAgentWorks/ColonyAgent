"""Experience management module - manages experience CRUD operations, merging, and persistence.

Based on XSkill's experience_manager.py, adapted for microbial image processing scenarios.
Uses LLMClient's embedding functionality for similarity computation.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# --------- Prompt Templates ---------

MERGE_EXPERIENCE_PROMPT = """\
You are an expert at consolidating microscopy image processing experiences.

Two experiences below are highly similar and should be merged into one concise, \
generalized experience.

IMPORTANT - Generalization Rules:
1. Replace specific thresholds (e.g., "blur_score < 0.2") with severity levels (e.g., "severe blur")
2. Keep only the degradation TYPE (blur/noise/contrast/illumination), not exact values
3. If both experiences have the same degradation type but different severity, use the more severe one
4. Merge actions if they target the same degradation type
5. Combine reasons to explain the general pattern, not specific case details

## Experience 1
- Condition: {cond1}
- Action: {action1}
- Reason: {reason1}

## Experience 2
- Condition: {cond2}
- Action: {action2}
- Reason: {reason2}

Output the merged, generalized experience as JSON:
```json
{{
    "condition": "General condition level (e.g., 'When blur is the dominant issue')",
    "action": "General action (e.g., 'Apply deblurring before detection')",
    "reason": "General reason explaining the pattern"
}}
```
"""


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: Vector a
        b: Vector b

    Returns:
        Cosine similarity [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ExperienceManager:
    """Experience manager - manages experience library CRUD operations and persistence."""

    def __init__(
        self,
        llm_client,
        save_dir: str,
        similarity_threshold: float = 0.85,
    ):
        """Initialize experience manager.

        Args:
            llm_client: LLMClient instance, used for merging experiences and generating embeddings.
            save_dir: Experience storage directory.
            similarity_threshold: Similarity threshold; triggers merge when exceeded.
        """
        self.experiences: Dict[str, Dict] = {}  # id -> experience
        self.llm = llm_client
        self.save_dir = save_dir
        self.similarity_threshold = similarity_threshold

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

    def add(self, experience: Dict) -> str:
        """Add new experience; merges with existing highly similar experience if found.

        Args:
            experience: Experience dict, must contain at least condition, action, reason.

        Returns:
            Final stored experience ID.
        """
        # Ensure id exists
        if "id" not in experience:
            experience["id"] = f"exp_{uuid.uuid4().hex[:8]}"

        exp_id = experience["id"]

        # Initialize metadata
        now = datetime.now().isoformat()
        experience.setdefault("created_at", now)
        experience.setdefault("updated_at", now)
        experience.setdefault("usage_count", 0)
        experience.setdefault("success_count", 0)
        experience.setdefault("success_rate", 0.0)
        experience.setdefault("source", "unknown")

        # Calculate new experience's embedding
        new_embedding = self._get_experience_embedding(experience)

        # Check if there is an existing experience with high similarity
        if new_embedding is not None:
            best_match_id, best_sim = self._find_most_similar(new_embedding)
            if best_match_id and best_sim >= self.similarity_threshold:
                logger.info(
                    f"Experience {exp_id} similar to {best_match_id} "
                    f"(sim={best_sim:.3f}), merging"
                )
                merged = self._merge_experiences(
                    self.experiences[best_match_id], experience
                )
                merged["id"] = best_match_id
                merged["updated_at"] = now
                merged["usage_count"] = (
                    self.experiences[best_match_id].get("usage_count", 0)
                    + experience.get("usage_count", 0)
                )
                # Recalculate embedding
                merged_emb = self._get_experience_embedding(merged)
                if merged_emb is not None:
                    merged["embedding"] = merged_emb.tolist()
                self.experiences[best_match_id] = merged
                return best_match_id

        # No similar experience, add directly
        if new_embedding is not None:
            experience["embedding"] = new_embedding.tolist()
        self.experiences[exp_id] = experience
        logger.info(f"Added new experience: {exp_id}")
        return exp_id

    def modify(self, exp_id: str, updated: Dict):
        """Modify existing experience.

        Args:
            exp_id: Experience ID to modify.
            updated: Dictionary of fields to update.
        """
        if exp_id not in self.experiences:
            logger.warning(f"Experience {exp_id} not found, cannot modify")
            return

        self.experiences[exp_id].update(updated)
        self.experiences[exp_id]["updated_at"] = datetime.now().isoformat()

        # If condition/action/reason is modified, recalculate embedding
        if any(k in updated for k in ("condition", "action", "reason")):
            emb = self._get_experience_embedding(self.experiences[exp_id])
            if emb is not None:
                self.experiences[exp_id]["embedding"] = emb.tolist()

        logger.info(f"Modified experience: {exp_id}")

    def remove(self, exp_id: str):
        """Delete experience.

        Args:
            exp_id: Experience ID to delete.
        """
        if exp_id in self.experiences:
            del self.experiences[exp_id]
            logger.info(f"Removed experience: {exp_id}")
        else:
            logger.warning(f"Experience {exp_id} not found, cannot remove")

    def update_stats(self, exp_id: str, successful: bool):
        """Update experience usage statistics.

        Args:
            exp_id: Experience ID.
            successful: Whether this usage was successful (verdict == "improved").
        """
        if exp_id not in self.experiences:
            return

        exp = self.experiences[exp_id]
        usage_count = exp.get("usage_count", 0) + 1
        old_success_count = exp.get("success_count", 0)
        new_success_count = old_success_count + (1 if successful else 0)
        success_rate = new_success_count / usage_count if usage_count > 0 else 0.0

        exp["usage_count"] = usage_count
        exp["success_count"] = new_success_count
        exp["success_rate"] = success_rate
        exp["updated_at"] = datetime.now().isoformat()

    def prune_low_quality(self, min_usage: int = 5, max_success_rate: float = 0.2):
        """Delete low-quality experiences.

        Deletes experiences where usage_count >= min_usage and success_rate < max_success_rate.

        Args:
            min_usage: Minimum usage count.
            max_success_rate: Maximum success rate threshold.
        """
        to_remove = []
        for exp_id, exp in self.experiences.items():
            if exp.get("usage_count", 0) >= min_usage and exp.get("success_rate", 1.0) < max_success_rate:
                to_remove.append(exp_id)

        for exp_id in to_remove:
            self.remove(exp_id)
            logger.info(f"Pruned low-quality experience: {exp_id}")

        if to_remove:
            logger.info(f"Pruned {len(to_remove)} low-quality experiences")

    def batch_merge(self, new_ops: List[Dict]):
        """Batch process experience update operations (from critique output).

        Args:
            new_ops: Operation list, each item format:
                {"action": "add"/"modify", "experience": {...}, "exp_id": "..." (for modify)}
        """
        if not new_ops:
            return

        add_count = 0
        modify_count = 0

        for op in new_ops:
            action = op.get("action", "add")
            experience = op.get("experience", {})

            if action == "add":
                self.add(experience)
                add_count += 1
            elif action == "modify":
                exp_id = op.get("exp_id", experience.get("id", ""))
                if exp_id and exp_id in self.experiences:
                    self.modify(exp_id, experience)
                    modify_count += 1
                else:
                    # Referenced ID doesn't exist, treat as add
                    self.add(experience)
                    add_count += 1

        logger.info(
            f"Batch merge complete: {add_count} added, {modify_count} modified, "
            f"library size: {len(self.experiences)}"
        )

    def get_all(self) -> List[Dict]:
        """Return all experiences.

        Returns:
            Experience list (copies without embedding field).
        """
        result = []
        for exp in self.experiences.values():
            exp_copy = {k: v for k, v in exp.items() if k != "embedding"}
            result.append(exp_copy)
        return result

    def save(self):
        """Save experience library to JSON file."""
        save_path = os.path.join(self.save_dir, "experiences.json")

        # Save without embedding (too large); embeddings will be recalculated on load
        serializable = {}
        for exp_id, exp in self.experiences.items():
            serializable[exp_id] = {
                k: v for k, v in exp.items() if k != "embedding"
            }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.experiences)} experiences to {save_path}")

    def load(self):
        """Load experience library from JSON file."""
        load_path = os.path.join(self.save_dir, "experiences.json")

        if not os.path.exists(load_path):
            logger.info(f"No experience file found at {load_path}, starting fresh")
            return

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.experiences = {}
        for exp_id, exp in data.items():
            exp["id"] = exp_id
            self.experiences[exp_id] = exp

        logger.info(f"Loaded {len(self.experiences)} experiences from {load_path}")

    def _compute_similarity(self, exp1: Dict, exp2: Dict) -> float:
        """Calculate text similarity between two experiences (using embedding).

        Args:
            exp1: Experience 1
            exp2: Experience 2

        Returns:
            Cosine similarity
        """
        emb1 = self._get_experience_embedding(exp1)
        emb2 = self._get_experience_embedding(exp2)

        if emb1 is None or emb2 is None:
            return 0.0

        return _cosine_similarity(emb1, emb2)

    def _merge_experiences(self, exp1: Dict, exp2: Dict) -> Dict:
        """Merge two similar experiences using LLM.

        Args:
            exp1: Experience 1
            exp2: Experience 2

        Returns:
            Merged experience dict
        """
        prompt = MERGE_EXPERIENCE_PROMPT.format(
            cond1=exp1.get("condition", ""),
            action1=exp1.get("action", ""),
            reason1=exp1.get("reason", ""),
            cond2=exp2.get("condition", ""),
            action2=exp2.get("action", ""),
            reason2=exp2.get("reason", ""),
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm.chat(messages, temperature=0.2)
            merged = self._parse_merge_response(response)
            if merged:
                # Preserve metadata
                result = dict(exp1)
                result["condition"] = merged.get("condition", exp1.get("condition", ""))
                result["action"] = merged.get("action", exp1.get("action", ""))
                result["reason"] = merged.get("reason", exp1.get("reason", ""))
                result["source"] = "merged"
                return result
        except Exception as e:
            logger.warning(f"LLM merge failed: {e}, keeping exp1")

        return dict(exp1)

    def _parse_merge_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from merge response.

        Args:
            response: LLM response text

        Returns:
            Parsed dict, or None
        """
        import re

        try:
            if "```json" in response:
                payload = response.split("```json")[-1].split("```")[0].strip()
            elif "```" in response:
                payload = response.split("```")[1].split("```")[0].strip()
            else:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                payload = match.group(0) if match else None

            if payload:
                return json.loads(payload)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse merge response: {e}")

        return None

    def _get_experience_embedding(self, experience: Dict) -> Optional[np.ndarray]:
        """Get experience's text embedding.

        Concatenates condition + action + reason into text, calls LLM embedding API.

        Args:
            experience: Experience dict

        Returns:
            Embedding vector, or None
        """
        # If there is already a cached embedding
        if "embedding" in experience and experience["embedding"]:
            return np.array(experience["embedding"], dtype=np.float32)

        text = self._experience_to_text(experience)
        if not text:
            return None

        try:
            embedding = self.llm.get_embedding(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return None

    def _experience_to_text(self, experience: Dict) -> str:
        """Convert experience to text representation for embedding.

        Args:
            experience: Experience dict

        Returns:
            Concatenated text
        """
        parts = []
        if experience.get("condition"):
            parts.append(f"Condition: {experience['condition']}")
        if experience.get("action"):
            parts.append(f"Action: {experience['action']}")
        if experience.get("reason"):
            parts.append(f"Reason: {experience['reason']}")
        return " | ".join(parts)

    def _find_most_similar(
        self, query_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Find the most similar experience to the query in existing experiences.

        Args:
            query_embedding: Query embedding

        Returns:
            (Most similar experience ID, similarity), or (None, 0.0) if no experiences exist
        """
        best_id = None
        best_sim = 0.0

        for exp_id, exp in self.experiences.items():
            emb = self._get_experience_embedding(exp)
            if emb is None:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_id = exp_id

        return best_id, best_sim
