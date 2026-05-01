"""Colony separation tool using watershed."""

import cv2
import numpy as np
from scipy import ndimage

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool
class ColonySeparationTool(BaseTool):
    name = "colony_separate"
    description = "Separate touching colonies using the watershed algorithm."
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["watershed"],
                "description": "Separation method.",
                "default": "watershed",
            },
            "min_distance": {
                "type": "integer",
                "description": "Minimum distance between local maxima in distance transform.",
                "default": 20,
            },
        },
        "required": [],
    }

    def call(self, image: np.ndarray, **params) -> ToolResult:
        method = params.get("method", "watershed")
        min_distance = int(params.get("min_distance", 20))

        try:
            if method == "watershed":
                result, meta = self._watershed(image, min_distance)
            else:
                return ToolResult(image=image, success=False, error=f"Unknown method: {method}")

            meta["method"] = method
            meta["min_distance"] = min_distance
            return ToolResult(image=result, metadata=meta)
        except Exception as e:
            return ToolResult(image=image, success=False, error=str(e))

    @staticmethod
    def _watershed(image: np.ndarray, min_distance: int):
        """Watershed-based colony separation."""
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal with morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background via dilation
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Distance transform and find local maxima as sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        # Find local maxima using scipy
        local_max = ndimage.maximum_filter(dist_transform, size=min_distance)
        local_max_mask = (dist_transform == local_max) & (dist_transform > 0.2 * dist_transform.max())
        local_max_mask = local_max_mask.astype(np.uint8)

        # Label markers
        num_labels, markers = cv2.connectedComponents(local_max_mask)

        # Mark background as 1, unknown as 0
        markers = markers + 1
        unknown = cv2.subtract(sure_bg, local_max_mask * 255)
        markers[unknown == 255] = 0

        # Apply watershed
        result = image.copy()
        markers = cv2.watershed(result, markers)

        # Draw separation lines (watershed boundaries are marked as -1)
        result[markers == -1] = [0, 0, 255]  # Red separation lines

        num_colonies = num_labels - 1  # Subtract background label

        return result, {
            "num_colonies": int(num_colonies),
            "num_labels": int(num_labels),
        }
