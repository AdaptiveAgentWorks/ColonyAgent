"""Sharpening / deblurring tool."""

import cv2
import numpy as np

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool
class SharpenTool(BaseTool):
    name = "sharpen"
    description = "Sharpen images using unsharp mask or Laplacian methods."
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["unsharp_mask", "laplacian"],
                "description": "Sharpening method.",
                "default": "unsharp_mask",
            },
            "strength": {
                "type": "number",
                "description": "Sharpening strength multiplier.",
                "default": 1.0,
            },
        },
        "required": [],
    }

    def call(self, image: np.ndarray, **params) -> ToolResult:
        method = params.get("method", "unsharp_mask")
        strength = float(params.get("strength", 1.0))

        try:
            if method == "unsharp_mask":
                result = self._unsharp_mask(image, strength)
            elif method == "laplacian":
                result = self._laplacian(image, strength)
            else:
                return ToolResult(image=image, success=False, error=f"Unknown method: {method}")

            return ToolResult(
                image=result,
                metadata={"method": method, "strength": strength},
            )
        except Exception as e:
            return ToolResult(image=image, success=False, error=str(e))

    @staticmethod
    def _unsharp_mask(image: np.ndarray, strength: float) -> np.ndarray:
        """Unsharp mask: original + strength * (original - blurred)."""
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        sharpened = cv2.addWeighted(
            image, 1.0 + strength, blurred, -strength, 0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def _laplacian(image: np.ndarray, strength: float) -> np.ndarray:
        """Laplacian sharpening: original - strength * laplacian."""
        lap = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = image.astype(np.float64) - strength * lap
        return np.clip(sharpened, 0, 255).astype(np.uint8)
