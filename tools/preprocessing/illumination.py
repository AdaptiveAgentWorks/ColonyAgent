"""Illumination correction tool."""

import cv2
import numpy as np

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool
class IlluminationCorrectionTool(BaseTool):
    name = "illumination_correct"
    description = "Correct uneven illumination using top-hat, retinex, or homomorphic filtering."
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["top_hat", "retinex", "homomorphic"],
                "description": "Illumination correction method.",
                "default": "retinex",
            },
            "kernel_size": {
                "type": "integer",
                "description": "Kernel size for filtering.",
                "default": 50,
            },
        },
        "required": [],
    }

    def call(self, image: np.ndarray, **params) -> ToolResult:
        method = params.get("method", "retinex")
        kernel_size = int(params.get("kernel_size", 50))

        try:
            if method == "top_hat":
                result = self._top_hat(image, kernel_size)
            elif method == "retinex":
                result = self._retinex(image, kernel_size)
            elif method == "homomorphic":
                result = self._homomorphic(image, kernel_size)
            else:
                return ToolResult(image=image, success=False, error=f"Unknown method: {method}")

            return ToolResult(
                image=result,
                metadata={"method": method, "kernel_size": kernel_size},
            )
        except Exception as e:
            return ToolResult(image=image, success=False, error=str(e))

    @staticmethod
    def _top_hat(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Morphological top-hat transform for uneven illumination correction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        if len(image.shape) == 3:
            # Apply correction per channel
            result = image.copy()
            for c in range(3):
                channel = image[:, :, c].astype(np.float32)
                bg = cv2.morphologyEx(image[:, :, c], cv2.MORPH_CLOSE, kernel).astype(np.float32)
                mean_bg = np.mean(bg)
                corrected = channel * (mean_bg / (bg + 1e-6))
                result[:, :, c] = np.clip(corrected, 0, 255).astype(np.uint8)
            return result
        else:
            bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel).astype(np.float32)
            mean_bg = np.mean(bg)
            corrected = gray.astype(np.float32) * (mean_bg / (bg + 1e-6))
            return np.clip(corrected, 0, 255).astype(np.uint8)

    @staticmethod
    def _retinex(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Single Scale Retinex (SSR) using Gaussian blur to estimate illumination."""
        # Ensure kernel_size is odd
        ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        img_float = image.astype(np.float64) + 1.0
        blur = cv2.GaussianBlur(img_float, (ks, ks), kernel_size / 3.0)
        retinex = np.log10(img_float) - np.log10(blur + 1.0)

        # Normalize to 0-255
        for c in range(retinex.shape[2]) if len(retinex.shape) == 3 else [None]:
            ch = retinex[:, :, c] if c is not None else retinex
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 0:
                if c is not None:
                    retinex[:, :, c] = (ch - ch_min) / (ch_max - ch_min) * 255
                else:
                    retinex = (ch - ch_min) / (ch_max - ch_min) * 255

        return retinex.astype(np.uint8)

    @staticmethod
    def _homomorphic(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Homomorphic filtering in the frequency domain."""
        if len(image.shape) == 3:
            # Process each channel
            channels = cv2.split(image)
            filtered = []
            for ch in channels:
                filtered.append(IlluminationCorrectionTool._homomorphic_single(ch, kernel_size))
            return cv2.merge(filtered)
        else:
            return IlluminationCorrectionTool._homomorphic_single(image, kernel_size)

    @staticmethod
    def _homomorphic_single(channel: np.ndarray, kernel_size: int) -> np.ndarray:
        """Homomorphic filtering on a single channel."""
        rows, cols = channel.shape
        # Take log
        img_log = np.log1p(channel.astype(np.float64))

        # DFT
        dft = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=(0, 1))

        # Create high-pass filter (Butterworth-like)
        crow, ccol = rows // 2, cols // 2
        d0 = kernel_size
        gamma_h, gamma_l, c_val = 1.5, 0.5, 1.0

        u = np.arange(rows).reshape(-1, 1) - crow
        v = np.arange(cols).reshape(1, -1) - ccol
        d_sq = u ** 2 + v ** 2
        h = (gamma_h - gamma_l) * (1.0 - np.exp(-c_val * d_sq / (2.0 * d0 ** 2 + 1e-6))) + gamma_l
        h = h[:, :, np.newaxis]

        # Apply filter
        dft_shift *= h
        dft_ishift = np.fft.ifftshift(dft_shift, axes=(0, 1))
        result = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Exp and normalize
        result = np.expm1(result)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)
