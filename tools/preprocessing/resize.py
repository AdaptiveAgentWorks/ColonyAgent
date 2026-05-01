"""
Tool-8: Image Resize/Crop Tool

Strategies for processing high-resolution images:
1. direct_resize: Directly resize to target size (simple and fast, may lose small objects)
2. sliding_window: Crop into multiple patches with sliding window (preserves details, suitable for dense small objects)
3. smart_resize: First resize to appropriate intermediate size + sharpening compensation (balances efficiency and accuracy)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool
class ResizeTool(BaseTool):
    name = "resize"
    description = (
        "Image resize/crop tool. Processes high-resolution images (e.g., 5120x5120), "
        "resizing them to dimensions suitable for detection model input. Supports three modes: "
        "direct resize, sliding window crop, and smart resize with sharpening compensation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["direct_resize", "sliding_window", "smart_resize"],
                "default": "smart_resize",
                "description": (
                    "Resize strategy: "
                    "direct_resize=direct resize; "
                    "sliding_window=slide window crop into multiple patches; "
                    "smart_resize=resize+sharpening compensation"
                ),
            },
            "target_size": {
                "type": "integer",
                "default": 640,
                "description": "Target size (square side length)",
            },
            "overlap": {
                "type": "number",
                "default": 0.2,
                "description": "Sliding window overlap ratio (sliding_window mode only)",
            },
            "sharpen_strength": {
                "type": "number",
                "default": 0.5,
                "description": "Sharpening compensation strength (smart_resize mode only)",
            },
        },
        "required": [],
    }

    def call(self, image: np.ndarray, **params) -> ToolResult:
        try:
            method = params.get("method", "smart_resize")
            target_size = params.get("target_size", 640)

            h, w = image.shape[:2]
            max_dim = max(h, w)

            # If image is already small enough, no processing needed
            if max_dim <= target_size * 1.5:
                return ToolResult(
                    image=image,
                    metadata={
                        "action": "skip",
                        "reason": f"Image size {w}x{h} already near target {target_size}",
                        "original_size": [w, h],
                    },
                    success=True,
                )

            scale_ratio = max_dim / target_size

            if method == "direct_resize":
                result = self._direct_resize(image, target_size)
                meta = {"method": "direct_resize", "scale_ratio": scale_ratio}

            elif method == "sliding_window":
                overlap = params.get("overlap", 0.2)
                result, patches_info = self._sliding_window(image, target_size, overlap)
                meta = {
                    "method": "sliding_window",
                    "num_patches": len(patches_info),
                    "patches": patches_info,
                    "scale_ratio": scale_ratio,
                }

            elif method == "smart_resize":
                sharpen_strength = params.get("sharpen_strength", 0.5)
                result = self._smart_resize(image, target_size, sharpen_strength)
                meta = {
                    "method": "smart_resize",
                    "sharpen_strength": sharpen_strength,
                    "scale_ratio": scale_ratio,
                }

            else:
                return ToolResult(
                    image=image,
                    metadata={"error": f"Unknown method: {method}"},
                    success=False,
                    error=f"Unknown method: {method}",
                )

            meta["original_size"] = [w, h]
            meta["output_size"] = [result.shape[1], result.shape[0]]

            return ToolResult(image=result, metadata=meta, success=True)

        except Exception as e:
            return ToolResult(
                image=image,
                metadata={"error": str(e)},
                success=False,
                error=str(e),
            )

    def _direct_resize(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Directly resize to target size, maintaining aspect ratio, padding with black edges."""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to square
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_y = (target_size - new_h) // 2
        pad_x = (target_size - new_w) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas

    def _sliding_window(self, image: np.ndarray, target_size: int,
                        overlap: float) -> Tuple[np.ndarray, List[Dict]]:
        """
        Sliding window crop. Returns stitched result image and patch information.
        Note: Returns original image (unchanged), patch info stored in metadata for subsequent detection.
        Actual detection needs to process each patch and merge results.
        """
        h, w = image.shape[:2]
        stride = int(target_size * (1 - overlap))
        patches_info = []

        y = 0
        while y < h:
            x = 0
            while x < w:
                x2 = min(x + target_size, w)
                y2 = min(y + target_size, h)
                x1 = max(0, x2 - target_size)
                y1 = max(0, y2 - target_size)

                patches_info.append({
                    "bbox": [x1, y1, x2, y2],
                    "index": len(patches_info),
                })

                x += stride
                if x2 >= w:
                    break

            y += stride
            if y2 >= h:
                break

        # For sliding window mode, return original image + patch info
        # Subsequent detection needs to crop and detect each patch based on patches_info
        return image, patches_info

    def _smart_resize(self, image: np.ndarray, target_size: int,
                      sharpen_strength: float) -> np.ndarray:
        """
        Smart resize: First resize with high-quality interpolation, then apply sharpening compensation to restore details.
        Suitable for moderate scaling (2-8x).
        """
        h, w = image.shape[:2]
        scale = target_size / max(h, w)

        # Multi-step resize (gradual downscaling, better quality than one-step)
        current = image
        current_scale = 1.0
        while current_scale * 0.5 > scale:
            ch, cw = current.shape[:2]
            current = cv2.resize(current, (cw // 2, ch // 2),
                                interpolation=cv2.INTER_AREA)
            current_scale *= 0.5

        # Final resize to target size
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Sharpening compensation
        if sharpen_strength > 0:
            blurred = cv2.GaussianBlur(resized, (0, 0), sigmaX=1.0)
            resized = cv2.addWeighted(resized, 1.0 + sharpen_strength,
                                      blurred, -sharpen_strength, 0)
            resized = np.clip(resized, 0, 255).astype(np.uint8)

        # Pad to square
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_y = (target_size - new_h) // 2
        pad_x = (target_size - new_w) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas
