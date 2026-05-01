"""LLM Client - Based on OpenAI compatible API"""

import os
import time
from typing import List, Dict, Optional, Any
from pathlib import Path

from openai import OpenAI
from loguru import logger

# Auto-load .env file
def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()


class LLMClient:
    """Unified LLM client supporting chat, function calling, and embedding.

    Uses the openai library, compatible with OpenAI API and any OpenAI-compatible endpoint.
    Built-in exponential backoff retry logic.
    """

    def __init__(self, config: dict):
        """Initialize client from config dictionary.

        Args:
            config: Config dictionary, supported keys:
                - model_name: Model name, default "gpt-4o"
                - api_key: API key, default from environment variable OPENAI_API_KEY
                - endpoint: API endpoint, default "https://api.openai.com/v1"
                - temperature: Sampling temperature, default 0.7
                - max_tokens: Maximum generation length, default 4096
                - max_retries: Maximum retry attempts, default 3
                - embedding: Embedding config sub-dictionary (model_name, api_key, endpoint, dimensions)
        """
        self.model_name = config.get("model_name", "gpt-4o")
        api_key_raw = config.get("api_key", "")
        # Support ${ENV_VAR} format environment variable reference
        if api_key_raw and api_key_raw.startswith("${") and api_key_raw.endswith("}"):
            env_var = api_key_raw[2:-1]
            api_key = os.environ.get(env_var, "")
        else:
            api_key = api_key_raw or os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        endpoint = config.get("endpoint", "https://api.openai.com/v1")
        self.client = OpenAI(api_key=api_key, base_url=endpoint)
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.max_retries = config.get("max_retries", 3)

        # Embedding configuration
        embedding_config = config.get("embedding", {})
        self.embedding_mode = embedding_config.get("mode", "local")  # "local" or "api"
        self.embedding_model_name = embedding_config.get("model_name", "all-MiniLM-L6-v2")
        self.embedding_dimensions = embedding_config.get("dimensions", None)  # DashScope support
        self._local_embedding_model = None  # Lazy loading

        if self.embedding_mode == "api":
            emb_api_key_raw = embedding_config.get("api_key", "")
            # Support ${ENV_VAR} format
            if emb_api_key_raw.startswith("${") and emb_api_key_raw.endswith("}"):
                env_var = emb_api_key_raw[2:-1]
                emb_api_key = os.environ.get(env_var, "")
            else:
                emb_api_key = emb_api_key_raw or api_key
            emb_endpoint = embedding_config.get("endpoint") or endpoint
            self.embedding_client = OpenAI(api_key=emb_api_key, base_url=emb_endpoint)
        else:
            self.embedding_client = None

    def _retry_call(self, func, *args, **kwargs) -> Any:
        """Retry call with exponential backoff.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function return value

        Raises:
            Last retry exception
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt  # 1s, 2s, 4s ...
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"API call failed after {self.max_retries} retries: {e}"
                    )
        raise last_exception

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Regular chat.

        Args:
            messages: Message list, format [{"role": "user", "content": "..."}]
            **kwargs: Extra parameters, can override temperature, max_tokens, model, etc.

        Returns:
            Model response text
        """
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        model = kwargs.pop("model", self.model_name)

        def _call():
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content or ""

        return self._retry_call(_call)

    def chat_with_tools(
        self, messages: List[Dict], tools: List[Dict], **kwargs
    ) -> dict:
        """Function calling mode chat.

        Args:
            messages: Message list
            tools: Tool definition list, OpenAI function calling format
            **kwargs: Extra parameters

        Returns:
            dict containing:
                - content: Model text response (if any)
                - tool_calls: Tool call list, each item contains name, arguments
                - raw_message: Raw message object
        """
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        model = kwargs.pop("model", self.model_name)
        tool_choice = kwargs.pop("tool_choice", "auto")

        def _call():
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            message = response.choices[0].message

            result = {
                "content": message.content or "",
                "tool_calls": [],
                "raw_message": message,
            }

            if message.tool_calls:
                for tc in message.tool_calls:
                    result["tool_calls"].append(
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    )

            return result

        return self._retry_call(_call)

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """Get text embedding vector.

        Supports two modes:
        - local: Use sentence-transformers local model (default, free, no API needed)
        - api: Use OpenAI-compatible embedding API

        Args:
            text: Input text
            **kwargs: Extra parameters

        Returns:
            Embedding vector (List[float])
        """
        if self.embedding_mode == "local":
            return self._get_local_embedding(text)
        else:
            model = kwargs.pop("model", self.embedding_model_name)

            def _call():
                params = {
                    "model": model,
                    "input": text,
                }
                # Add dimensions parameter (DashScope support)
                if self.embedding_dimensions:
                    params["dimensions"] = self.embedding_dimensions
                response = self.embedding_client.embeddings.create(**params)
                return response.data[0].embedding

            return self._retry_call(_call)

    def _get_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local sentence-transformers model."""
        if self._local_embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local embedding model: {self.embedding_model_name}")
                self._local_embedding_model = SentenceTransformer(self.embedding_model_name)
                self._local_embedding_model = self._local_embedding_model.to("cuda:1")
                logger.info(f"Local embedding model loaded to cuda:1")
            except ImportError:
                raise ImportError(
                    "Local embedding requires sentence-transformers library. "
                    "Please run: pip install sentence-transformers"
                )
        embedding = self._local_embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
