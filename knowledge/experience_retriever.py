"""Experience retrieval module - retrieves related experiences by semantic or quality features.

Based on XSkill's experience_retriever.py, uses numpy to implement cosine similarity retrieval.
"""

from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from .experience_manager import ExperienceManager, _cosine_similarity


class ExperienceRetriever:
    """Experience retriever - supports semantic retrieval and quality vector retrieval."""

    def __init__(
        self,
        llm_client,
        experience_manager: ExperienceManager,
        top_k: int = 3,
        min_similarity: float = 0.6,
    ):
        """Initialize retriever.

        Args:
            llm_client: LLMClient instance for generating embeddings.
            experience_manager: ExperienceManager instance.
            top_k: Default number of top-k results to return.
            min_similarity: Minimum similarity threshold.
        """
        self.llm = llm_client
        self.manager = experience_manager
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.embeddings_cache: Dict[str, np.ndarray] = {}

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve top-k experiences by text semantic similarity.

        Converts query to embedding, calculates cosine similarity with all experience embeddings,
        and returns the top-k experiences with highest similarity.

        Args:
            query: Query text (e.g., current image's quality description or problem description).
            top_k: Number of results to return, defaults to self.top_k.

        Returns:
            List of matched experiences, each containing experience dict and similarity score.
        """
        if top_k is None:
            top_k = self.top_k

        if not self.manager.experiences:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.warning("Failed to get query embedding, returning empty results")
            return []

        # Build index and retrieve
        self._build_index()

        # Calculate similarity
        results = []
        for exp_id, exp in self.manager.experiences.items():
            if exp_id not in self.embeddings_cache:
                continue

            exp_emb = self.embeddings_cache[exp_id]
            sim = _cosine_similarity(query_embedding, exp_emb)

            if sim >= self.min_similarity:
                exp_copy = {k: v for k, v in exp.items() if k != "embedding"}
                exp_copy["_similarity"] = sim
                results.append(exp_copy)

        # Sort by similarity in descending order
        results.sort(key=lambda x: x["_similarity"], reverse=True)

        # Return top-k
        top_results = results[:top_k]

        # Filter low-quality experiences: do not return experiences with usage_count >= 3 and success_rate < 0.3
        filtered_results = [
            r for r in top_results
            if not (r.get("usage_count", 0) >= 3 and r.get("success_rate", 1.0) < 0.3)
        ]

        if filtered_results:
            logger.info(
                f"Retrieved {len(filtered_results)} experiences (filtered {len(top_results) - len(filtered_results)} low-quality)"
            )
        return filtered_results

    def retrieve_by_quality(
        self, quality_vector: np.ndarray, top_k: int = None
    ) -> List[Dict]:
        """Retrieve related experiences by image quality feature vector.

        Matches quality vector with quality conditions recorded in each experience.
        Uses structured queries instead of vague text to improve retrieval precision.

        Args:
            quality_vector: 5-dimensional quality feature vector
                [blur, brightness, contrast, noise, color_bias]
            top_k: Number of results to return.

        Returns:
            List of matched experiences.
        """
        dim_names = ["blur", "brightness", "contrast", "noise", "color_bias"]

        # Build structured query with specific values, not just vague descriptions
        quality_conditions = []
        for name, val in zip(dim_names, quality_vector):
            if val < 0.3:
                quality_conditions.append(f"{name}:severity=critical,value={val:.2f}")
            elif val < 0.5:
                quality_conditions.append(f"{name}:severity=low,value={val:.2f}")
            elif val < 0.7:
                quality_conditions.append(f"{name}:severity=moderate,value={val:.2f}")
            else:
                quality_conditions.append(f"{name}:severity=good,value={val:.2f}")

        # Build precise query, clearly indicating degradation types that need to be handled
        deg_types = []
        if quality_vector[0] < 0.5:  # blur
            deg_types.append("deblurring or sharpening")
        if quality_vector[2] < 0.5:  # contrast
            deg_types.append("contrast enhancement")
        if quality_vector[3] < 0.5:  # noise
            deg_types.append("denoising")
        if quality_vector[4] < 0.5:  # color bias
            deg_types.append("white balance correction")

        processing_needs = ", ".join(deg_types) if deg_types else "direct detection"

        query = (
            f"Image quality: {', '.join(quality_conditions)}. "
            f"Needs: {processing_needs}. "
            f"What preprocessing tools should be applied?"
        )

        return self.retrieve(query, top_k=top_k)

    def _build_index(self):
        """Build/update embedding index.

        Iterates through all experiences to ensure each has an embedding cache.
        """
        for exp_id, exp in self.manager.experiences.items():
            if exp_id in self.embeddings_cache:
                continue

            # First check if experience already has an embedding
            if "embedding" in exp and exp["embedding"]:
                self.embeddings_cache[exp_id] = np.array(
                    exp["embedding"], dtype=np.float32
                )
                continue

            # Generate embedding
            text = self.manager._experience_to_text(exp)
            if text:
                emb = self._get_embedding(text)
                if emb is not None:
                    self.embeddings_cache[exp_id] = emb

        # Clean up cache for deleted experiences
        cached_ids = set(self.embeddings_cache.keys())
        current_ids = set(self.manager.experiences.keys())
        for removed_id in cached_ids - current_ids:
            del self.embeddings_cache[removed_id]

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding with caching.

        Args:
            text: Input text.

        Returns:
            Embedding vector, or None.
        """
        # Check cache (using text hash as key)
        import hashlib

        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_key = f"_text_{text_hash}"

        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]

        try:
            embedding = self.llm.get_embedding(text)
            emb_array = np.array(embedding, dtype=np.float32)
            self.embeddings_cache[cache_key] = emb_array
            return emb_array
        except Exception as e:
            logger.warning(f"Failed to get embedding for text: {e}")
            return None

    def invalidate_cache(self, exp_id: str = None):
        """Clear embedding cache.

        Args:
            exp_id: If specified, only clear cache for that specific experience; otherwise clear all.
        """
        if exp_id:
            self.embeddings_cache.pop(exp_id, None)
        else:
            self.embeddings_cache.clear()
            logger.info("Cleared all embedding caches")
