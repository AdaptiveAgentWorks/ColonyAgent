"""CLAHE contrast enhancement tool."""

import cv2
import numpy as np

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool
class CLAHETool(BaseTool):
    name = "clahe_enhance"
    description = "Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel in LAB color space."
    parameters = {
        "type": "object",
        "properties": {
            "clip_limit": {
                "type": "number",
                "description": "Contrast clip limit for CLAHE.",
                "default": 2.0,
            },
            "tile_size": {
                "type": "integer",
                "description": "Tile grid size for CLAHE.",
                "default": 8,
            },
        },
        "required": [],
    }

    def call(self, image: np.ndarray, **params) -> ToolResult:
        clip_limit = float(params.get("clip_limit", 2.0))
        tile_size = int(params.get("tile_size", 8))

        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(tile_size, tile_size),
            )
            l_enhanced = clahe.apply(l)

            # Merge and convert back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            return ToolResult(
                image=result,
                metadata={
                    "clip_limit": clip_limit,
                    "tile_size": tile_size,
                },
            )
        except Exception as e:
            return ToolResult(image=image, success=False, error=str(e))
