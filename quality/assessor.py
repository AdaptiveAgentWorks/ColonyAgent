"""Image Quality Assessor - 6-dimensional quality assessment.

Extended from MicroAgent's assess_quality method, adding noise and color bias assessment dimensions.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualityReport:
    """Image quality assessment report."""

    blur_score: float  # 0-1, higher is sharper
    brightness_score: float  # 0-1, closer to 1 is better
    contrast_score: float  # 0-1, higher is better contrast
    noise_score: float  # 0-1, higher means less noise
    color_bias_score: float  # 0-1, higher means less color bias
    overall_score: float  # Weighted composite score
    raw_metrics: dict = field(default_factory=dict)  # Raw metric values


# Default weights
DEFAULT_WEIGHTS = {
    "blur": 0.25,
    "brightness": 0.20,
    "contrast": 0.20,
    "noise": 0.20,
    "color_bias": 0.15,
}

# Default thresholds (calibrated for AGAR microbial colony dataset)
DEFAULT_THRESHOLDS = {
    "blur_threshold": 25.0,  # Laplacian variance reaching this value is considered fully sharp (AGAR clean images ≈20)
    "brightness_min": 30,  # Minimum acceptable brightness
    "brightness_max": 220,  # Maximum acceptable brightness
    "contrast_good": 50.0,  # Standard deviation reaching this value is considered good contrast
    "noise_lap_var_max": 100.0,  # Laplacian variance exceeding this indicates noise (clean images ≈20-70, Gaussian noise >500)
    "color_bias_var_max": 400.0,  # Upper limit for channel mean variance
}


class ImageQualityAssessor:
    """Image quality assessor.

    Supports 6-dimensional quality assessment: blur, brightness, contrast, noise, color bias.
    All metrics normalized to 0-1 range, supports custom weights and thresholds.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize assessor.

        Args:
            config: Config dictionary, optional keys:
                - weights: Dimension weights dictionary (blur, brightness, contrast, noise, color_bias)
                - thresholds: Threshold dictionary
        """
        config = config or {}

        # Load weights
        weights_cfg = config.get("weights", {})
        self.weights = {
            "blur": weights_cfg.get("blur", DEFAULT_WEIGHTS["blur"]),
            "brightness": weights_cfg.get("brightness", DEFAULT_WEIGHTS["brightness"]),
            "contrast": weights_cfg.get("contrast", DEFAULT_WEIGHTS["contrast"]),
            "noise": weights_cfg.get("noise", DEFAULT_WEIGHTS["noise"]),
            "color_bias": weights_cfg.get("color_bias", DEFAULT_WEIGHTS["color_bias"]),
        }

        # Load thresholds
        thresholds_cfg = config.get("thresholds", {})
        self.blur_threshold = thresholds_cfg.get(
            "blur_threshold", DEFAULT_THRESHOLDS["blur_threshold"]
        )
        self.brightness_min = thresholds_cfg.get(
            "brightness_min", DEFAULT_THRESHOLDS["brightness_min"]
        )
        self.brightness_max = thresholds_cfg.get(
            "brightness_max", DEFAULT_THRESHOLDS["brightness_max"]
        )
        self.contrast_good = thresholds_cfg.get(
            "contrast_good", DEFAULT_THRESHOLDS["contrast_good"]
        )
        self.noise_lap_var_max = thresholds_cfg.get(
            "noise_lap_var_max", DEFAULT_THRESHOLDS["noise_lap_var_max"]
        )
        self.color_bias_var_max = thresholds_cfg.get(
            "color_bias_var_max", DEFAULT_THRESHOLDS["color_bias_var_max"]
        )

    def assess(self, image: np.ndarray) -> QualityReport:
        """Full quality assessment.

        Args:
            image: Input image in BGR format (np.ndarray)

        Returns:
            QualityReport quality assessment report
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Per-dimension assessment
        blur_score, laplacian_var = self._assess_blur(gray)
        brightness_score, mean_brightness = self._assess_brightness(gray)
        contrast_score, std_dev = self._assess_contrast(gray)
        noise_score, noise_level = self._assess_noise(gray)
        color_bias_score, channel_means = self._assess_color_bias(image)

        # Weighted composite score
        overall_score = (
            self.weights["blur"] * blur_score
            + self.weights["brightness"] * brightness_score
            + self.weights["contrast"] * contrast_score
            + self.weights["noise"] * noise_score
            + self.weights["color_bias"] * color_bias_score
        )

        raw_metrics = {
            "laplacian_var": float(laplacian_var),
            "mean_brightness": float(mean_brightness),
            "std_dev": float(std_dev),
            "noise_level": float(noise_level),
            "channel_means": [float(v) for v in channel_means],
        }

        return QualityReport(
            blur_score=blur_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            noise_score=noise_score,
            color_bias_score=color_bias_score,
            overall_score=overall_score,
            raw_metrics=raw_metrics,
        )

    def _assess_blur(self, gray: np.ndarray) -> tuple:
        """Assess blur.

        Uses Laplacian variance to measure image sharpness.

        Args:
            gray: Grayscale image

        Returns:
            (score, laplacian_var): higher score means sharper
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        score = min(laplacian_var / self.blur_threshold, 1.0)
        return float(score), float(laplacian_var)

    def _assess_brightness(self, gray: np.ndarray) -> tuple:
        """Assess brightness.

        Brightness in range [brightness_min, brightness_max] is optimal.

        Args:
            gray: Grayscale image

        Returns:
            (score, mean_brightness): score closer to 1 is better
        """
        mean_brightness = float(np.mean(gray))

        if self.brightness_min <= mean_brightness <= self.brightness_max:
            score = 1.0
        elif mean_brightness < self.brightness_min:
            score = mean_brightness / self.brightness_min if self.brightness_min > 0 else 0.0
        else:
            denom = 255 - self.brightness_max
            score = 1.0 - (mean_brightness - self.brightness_max) / denom if denom > 0 else 0.0

        score = max(0.0, min(1.0, score))
        return score, mean_brightness

    def _assess_contrast(self, gray: np.ndarray) -> tuple:
        """Assess contrast.

        Uses grayscale standard deviation to measure contrast.

        Args:
            gray: Grayscale image

        Returns:
            (score, std_dev): higher score means better contrast
        """
        std_dev = float(np.std(gray))
        score = min(std_dev / self.contrast_good, 1.0)
        return score, std_dev

    def _assess_noise(self, gray: np.ndarray) -> tuple:
        """Assess noise level.

        Uses Laplacian variance to detect noise:
        - Clean images have low Laplacian variance (AGAR ~15-70)
        - Noisy images have abnormally high Laplacian variance (Gaussian noise > 1000)
        - Blurry images also have low Laplacian variance, but blur is handled by the blur metric

        Args:
            gray: Grayscale image

        Returns:
            (score, noise_level): higher score means less noise
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())

        # Laplacian variance exceeding threshold indicates noise
        # Clean images lap_var ≈ 15-70, Gaussian noise lap_var > 1000
        if lap_var <= self.noise_lap_var_max:
            # Within normal range, no significant noise
            score = 1.0
        else:
            # Exceeding threshold, higher noise means lower score
            score = max(0.0, 1.0 - (lap_var - self.noise_lap_var_max) / (self.noise_lap_var_max * 10))

        return score, lap_var

    def _assess_color_bias(self, image: np.ndarray) -> tuple:
        """Assess color bias.

        Calculates variance of BGR three-channel means, higher variance means more severe color bias.

        Args:
            image: BGR format image (returns full score if grayscale)

        Returns:
            (score, channel_means): higher score means less color bias
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return 1.0, [0.0, 0.0, 0.0]

        channel_means = [float(np.mean(image[:, :, c])) for c in range(3)]
        channel_var = float(np.var(channel_means))

        # Normalize: higher variance -> lower score
        score = max(0.0, 1.0 - channel_var / self.color_bias_var_max)
        return score, channel_means

    def to_feature_vector(self, report: QualityReport) -> np.ndarray:
        """Convert quality metrics to 5-dimensional vector for Experience retrieval.

        Args:
            report: Quality assessment report

        Returns:
            5-dimensional numpy array [blur, brightness, contrast, noise, color_bias]
        """
        return np.array(
            [
                report.blur_score,
                report.brightness_score,
                report.contrast_score,
                report.noise_score,
                report.color_bias_score,
            ],
            dtype=np.float32,
        )
