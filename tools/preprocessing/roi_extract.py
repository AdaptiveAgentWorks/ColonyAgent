"""ROI extraction tool."""

import cv2
import numpy as np

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool
class ROIExtractTool(BaseTool):
    name = "roi_extract"
    description = "Extract region of interest using Hough circle detection or contour detection."
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["hough", "contour"],
                "description": "ROI detection method.",
                "default": "hough",
            },
            "margin": {
                "type": "integer",
                "description": "Margin in pixels around detected ROI.",
                "default": 10,
            },
        },
        "required": [],
    }

    def call(self, image: np.ndarray, **params) -> ToolResult:
        method = params.get("method", "hough")
        margin = int(params.get("margin", 10))

        try:
            if method == "hough":
                result, meta = self._hough_extract(image, margin)
            elif method == "contour":
                result, meta = self._contour_extract(image, margin)
            else:
                return ToolResult(image=image, success=False, error=f"Unknown method: {method}")

            meta["method"] = method
            meta["margin"] = margin
            return ToolResult(image=result, metadata=meta)
        except Exception as e:
            return ToolResult(image=image, success=False, error=str(e))

    @staticmethod
    def _hough_extract(image: np.ndarray, margin: int):
        """Detect petri dish boundary via Hough circle transform and crop."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        h, w = gray.shape

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(h, w) // 2,
            param1=100,
            param2=50,
            minRadius=min(h, w) // 4,
            maxRadius=min(h, w) // 2,
        )

        if circles is None:
            return image, {"detected": False, "reason": "no_circle_found"}

        # Take the largest circle
        circles = np.uint16(np.around(circles))
        best = max(circles[0], key=lambda c: c[2])
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])

        # Create circular mask and crop bounding box
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)

        x1 = max(cx - r - margin, 0)
        y1 = max(cy - r - margin, 0)
        x2 = min(cx + r + margin, w)
        y2 = min(cy + r + margin, h)
        cropped = masked[y1:y2, x1:x2]

        return cropped, {
            "detected": True,
            "center": [cx, cy],
            "radius": int(r),
            "bbox": [x1, y1, x2, y2],
        }

    @staticmethod
    def _contour_extract(image: np.ndarray, margin: int):
        """Detect ROI via largest contour and crop."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image, {"detected": False, "reason": "no_contour_found"}

        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        h, w = image.shape[:2]

        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + cw + margin, w)
        y2 = min(y + ch + margin, h)
        cropped = image[y1:y2, x1:x2]

        return cropped, {
            "detected": True,
            "bbox": [x1, y1, x2, y2],
            "contour_area": float(cv2.contourArea(largest)),
        }
