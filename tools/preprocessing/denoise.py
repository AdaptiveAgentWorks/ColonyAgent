"""Denoising tool."""

import cv2
import numpy as np

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool
class DenoiseTool(BaseTool):
    name = "denoise"
    description = "Denoise images using Non-Local Means, median, or Gaussian filtering."
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["nlm", "median", "gaussian"],
                "description": "Denoising method.",
                "default": "nlm",
            },
            "strength": {
                "type": "integer",
                "description": "Denoising strength (h for NLM, kernel size for median/gaussian).",
                "default": 10,
            },
        },
        "required": [],
    }

    def call(self, image: np.ndarray, **params) -> ToolResult:
        method = params.get("method", "nlm")
        strength = int(params.get("strength", 10))

        try:
            if method == "nlm":
                result = cv2.fastNlMeansDenoisingColored(
                    image, None, strength, strength, 7, 21
                )
            elif method == "median":
                ksize = strength if strength % 2 == 1 else strength + 1
                result = cv2.medianBlur(image, ksize)
            elif method == "gaussian":
                ksize = strength if strength % 2 == 1 else strength + 1
                result = cv2.GaussianBlur(image, (ksize, ksize), 0)
            else:
                return ToolResult(image=image, success=False, error=f"Unknown method: {method}")

            return ToolResult(
                image=result,
                metadata={"method": method, "strength": strength},
            )
        except Exception as e:
            return ToolResult(image=image, success=False, error=str(e))
