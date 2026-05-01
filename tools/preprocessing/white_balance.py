"""White balance correction tool."""

import cv2
import numpy as np

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool
class WhiteBalanceTool(BaseTool):
    name = "white_balance"
    description = "Correct white balance using gray world or white patch assumption."
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["gray_world", "white_patch"],
                "description": "White balance correction method.",
                "default": "gray_world",
            },
        },
        "required": [],
    }

    def call(self, image: np.ndarray, **params) -> ToolResult:
        method = params.get("method", "gray_world")

        try:
            if method == "gray_world":
                result = self._gray_world(image)
            elif method == "white_patch":
                result = self._white_patch(image)
            else:
                return ToolResult(image=image, success=False, error=f"Unknown method: {method}")

            return ToolResult(image=result, metadata={"method": method})
        except Exception as e:
            return ToolResult(image=image, success=False, error=str(e))

    @staticmethod
    def _gray_world(image: np.ndarray) -> np.ndarray:
        """Gray world assumption: adjust each channel so that the mean is gray."""
        img = image.astype(np.float64)
        avg = img.mean()
        result = np.zeros_like(img)
        for c in range(3):
            ch_mean = img[:, :, c].mean()
            if ch_mean > 0:
                result[:, :, c] = img[:, :, c] * (avg / ch_mean)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _white_patch(image: np.ndarray) -> np.ndarray:
        """White patch assumption: scale each channel by the maximum value."""
        img = image.astype(np.float64)
        result = np.zeros_like(img)
        for c in range(3):
            ch_max = img[:, :, c].max()
            if ch_max > 0:
                result[:, :, c] = img[:, :, c] * (255.0 / ch_max)
        return np.clip(result, 0, 255).astype(np.uint8)
