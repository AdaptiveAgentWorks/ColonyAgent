"""
Base Tool Class - Defines the interface for all tools.

All tools must inherit BaseTool and implement the `call` method.
ToolResult is the standardised return type for every tool invocation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import numpy as np


@dataclass
class ToolResult:
    """Standardised result returned by every tool."""

    image: np.ndarray                                      # Processed image (detection tools return original)
    detections: List[Dict] = field(default_factory=list)   # Detection results
    metadata: Dict = field(default_factory=dict)           # Additional information
    success: bool = True
    error: Optional[str] = None


class BaseTool(ABC):
    """Base class for all tools. All tools must inherit this class and implement the call method."""

    name: str = ""           # Tool name
    description: str = ""    # Tool description
    parameters: dict = {}    # Parameters definition (JSON Schema format)

    def __init__(self, config: Dict = None):
        """
        Initialize the tool.

        Args:
            config: Tool configuration dictionary.
        """
        self.config = config or {}
        if not self.name:
            raise ValueError("Tool must have a name")

    @abstractmethod
    def call(self, image: np.ndarray, **params) -> ToolResult:
        """
        Execute the tool on an image.

        Args:
            image: Input image as a numpy array (H, W, C).
            **params: Tool-specific parameters.

        Returns:
            ToolResult with the processed image and metadata.
        """
        raise NotImplementedError

    def validate_params(self, params: dict) -> bool:
        """
        Validate parameters against the JSON Schema definition.

        Args:
            params: Parameter dictionary.

        Returns:
            Whether validation passed.
        """
        required = self.parameters.get("required", [])
        return all(key in params for key in required)

    def get_schema(self) -> dict:
        """Return an OpenAI function-calling compatible schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
