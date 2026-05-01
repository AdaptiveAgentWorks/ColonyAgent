"""
Image degradation module: Applies various quality defects to clean images to generate degraded data for testing the preprocessing framework.

Supported degradation types:
1. Gaussian noise (gaussian_noise)
2. Salt and pepper noise (salt_pepper_noise)
3. Brightness reduction (darken)
4. Uneven illumination (uneven_illumination)
5. Gaussian blur (gaussian_blur)
6. Motion blur (motion_blur)
7. Color bias (color_bias)
8. Low contrast (low_contrast)
9. Random rotation (rotation)
10. Combined degradation (combined)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import random
import os
import json
from copy import deepcopy


@dataclass
class DegradationRecord:
    """Record of applied degradation operations."""
    degradation_type: str
    params: Dict
    severity: str  # "mild", "moderate", "severe"


class ImageDegrader:
    """Image degrader: Applies various quality defects to clean images."""

    SEVERITY_LEVELS = {
        "mild": 0,
        "moderate": 1,
        "severe": 2,
    }

    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    # ==================== Single Degradation ====================

    def add_gaussian_noise(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, DegradationRecord]:
        """Add Gaussian noise."""
        sigma_map = {"mild": 15, "moderate": 30, "severe": 50}
        sigma = sigma_map[severity]

        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        record = DegradationRecord("gaussian_noise", {"sigma": sigma}, severity)
        return noisy, record

    def add_salt_pepper_noise(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, DegradationRecord]:
        """Add salt and pepper noise."""
        ratio_map = {"mild": 0.01, "moderate": 0.03, "severe": 0.06}
        ratio = ratio_map[severity]

        noisy = image.copy()
        h, w = image.shape[:2]
        num_pixels = int(h * w * ratio)

        # Salt noise (white pixels)
        coords = [np.random.randint(0, i, num_pixels) for i in [h, w]]
        noisy[coords[0], coords[1]] = 255

        # Pepper noise (black pixels)
        coords = [np.random.randint(0, i, num_pixels) for i in [h, w]]
        noisy[coords[0], coords[1]] = 0

        record = DegradationRecord("salt_pepper_noise", {"ratio": ratio}, severity)
        return noisy, record

    def darken(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, DegradationRecord]:
        """Reduce brightness."""
        factor_map = {"mild": 0.6, "moderate": 0.4, "severe": 0.2}
        factor = factor_map[severity]

        darkened = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        record = DegradationRecord("darken", {"factor": factor}, severity)
        return darkened, record

    def uneven_illumination(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, DegradationRecord]:
        """Simulate uneven illumination (petri dish center is bright, edges are dark)."""
        strength_map = {"mild": 0.3, "moderate": 0.5, "severe": 0.7}
        strength = strength_map[severity]

        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2

        # Generate radial gradient mask
        y, x = np.ogrid[:h, :w]
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Farther distance means darker
        mask = 1.0 - strength * (dist / max_dist)
        mask = np.clip(mask, 0.2, 1.0)

        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]

        result = np.clip(image.astype(np.float32) * mask, 0, 255).astype(np.uint8)

        record = DegradationRecord("uneven_illumination", {"strength": strength}, severity)
        return result, record

    def gaussian_blur(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, DegradationRecord]:
        """Gaussian blur."""
        ksize_map = {"mild": 5, "moderate": 11, "severe": 21}
        ksize = ksize_map[severity]

        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

        record = DegradationRecord("gaussian_blur", {"kernel_size": ksize}, severity)
        return blurred, record

    def motion_blur(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, DegradationRecord]:
        """Motion blur."""
        ksize_map = {"mild": 7, "moderate": 15, "severe": 25}
        ksize = ksize_map[severity]

        # Generate motion blur kernel (random direction)
        angle = np.random.uniform(0, 180)
        kernel = np.zeros((ksize, ksize))
        center = ksize // 2
        # Horizontal blur kernel, then rotate
        kernel[center, :] = 1.0
        kernel = kernel / ksize

        M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        kernel = kernel / (kernel.sum() + 1e-8)

        blurred = cv2.filter2D(image, -1, kernel)

        record = DegradationRecord("motion_blur", {"kernel_size": ksize, "angle": float(angle)}, severity)
        return blurred, record

    def color_bias(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, DegradationRecord]:
        """Add color bias (simulating different light sources)."""
        if len(image.shape) != 3:
            return image.copy(), DegradationRecord("color_bias", {"skipped": True}, severity)

        bias_map = {
            "mild": 20,
            "moderate": 40,
            "severe": 60,
        }
        max_bias = bias_map[severity]

        # Randomly select channel and direction for bias
        channel = np.random.randint(0, 3)  # B, G, R
        bias = np.random.randint(max_bias // 2, max_bias)
        direction = np.random.choice([-1, 1])

        result = image.astype(np.float32)
        result[:, :, channel] = np.clip(result[:, :, channel] + direction * bias, 0, 255)
        result = result.astype(np.uint8)

        channel_names = ["blue", "green", "red"]
        record = DegradationRecord("color_bias", {
            "channel": channel_names[channel],
            "bias": int(direction * bias),
        }, severity)
        return result, record

    def low_contrast(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, DegradationRecord]:
        """Reduce contrast (histogram compression)."""
        factor_map = {"mild": 0.6, "moderate": 0.4, "severe": 0.25}
        factor = factor_map[severity]

        mean = image.mean()
        result = np.clip(mean + (image.astype(np.float32) - mean) * factor, 0, 255).astype(np.uint8)

        record = DegradationRecord("low_contrast", {"factor": factor}, severity)
        return result, record

    def rotate(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, DegradationRecord]:
        """Random rotation."""
        angle_map = {
            "mild": (-15, 15),
            "moderate": (-45, 45),
            "severe": (-180, 180),
        }
        angle_range = angle_map[severity]
        angle = np.random.uniform(*angle_range)

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate rotated boundaries
        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int(h * sin_val + w * cos_val)
        new_h = int(h * cos_val + w * sin_val)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))

        record = DegradationRecord("rotation", {"angle": float(angle)}, severity)
        return rotated, record

    # ==================== Combined Degradation ====================

    def random_single(self, image: np.ndarray, severity: str = "moderate") -> Tuple[np.ndarray, List[DegradationRecord]]:
        """Randomly apply one degradation."""
        methods = [
            self.add_gaussian_noise, self.add_salt_pepper_noise,
            self.darken, self.uneven_illumination,
            self.gaussian_blur, self.motion_blur,
            self.color_bias, self.low_contrast, self.rotate,
        ]
        method = random.choice(methods)
        result, record = method(image, severity)
        return result, [record]

    def random_combined(self, image: np.ndarray, num_degradations: int = 2,
                        severity: str = "moderate") -> Tuple[np.ndarray, List[DegradationRecord]]:
        """Randomly combine multiple degradations."""
        methods = [
            self.add_gaussian_noise, self.add_salt_pepper_noise,
            self.darken, self.uneven_illumination,
            self.gaussian_blur, self.color_bias,
            self.low_contrast,
        ]
        # Exclude rotation (changes image size, affects annotations) and motion_blur (redundant with gaussian_blur)

        selected = random.sample(methods, min(num_degradations, len(methods)))
        records = []
        result = image.copy()

        for method in selected:
            result, record = method(result, severity)
            records.append(record)

        return result, records

    # ==================== Predefined Degradation Scenarios ====================

    def create_scenario(self, image: np.ndarray, scenario: str) -> Tuple[np.ndarray, List[DegradationRecord]]:
        """
        Predefined degradation scenarios simulating real laboratory conditions.

        Scenario list:
        - "dark_lab": Dark room environment (dark + noise)
        - "fluorescent": Fluorescent lighting (color bias + uneven illumination)
        - "phone_capture": Phone capture (blur + rotation + low contrast)
        - "old_equipment": Old equipment (noise + low contrast + blur)
        - "edge_lighting": Edge lighting (uneven illumination + reflection)
        """
        records = []
        result = image.copy()

        if scenario == "dark_lab":
            result, r = self.darken(result, "severe")
            records.append(r)
            result, r = self.add_gaussian_noise(result, "moderate")
            records.append(r)

        elif scenario == "fluorescent":
            result, r = self.color_bias(result, "moderate")
            records.append(r)
            result, r = self.uneven_illumination(result, "mild")
            records.append(r)

        elif scenario == "phone_capture":
            result, r = self.gaussian_blur(result, "mild")
            records.append(r)
            result, r = self.low_contrast(result, "mild")
            records.append(r)

        elif scenario == "old_equipment":
            result, r = self.add_gaussian_noise(result, "moderate")
            records.append(r)
            result, r = self.low_contrast(result, "moderate")
            records.append(r)
            result, r = self.gaussian_blur(result, "mild")
            records.append(r)

        elif scenario == "edge_lighting":
            result, r = self.uneven_illumination(result, "severe")
            records.append(r)
            result, r = self.darken(result, "mild")
            records.append(r)

        else:
            raise ValueError(f"Unknown scenario: {scenario}. "
                           f"Available: dark_lab, fluorescent, phone_capture, old_equipment, edge_lighting")

        return result, records


# ==================== Batch Dataset Generation ====================

class DegradedDatasetGenerator:
    """
    Generate degraded dataset from clean dataset.

    Generates multiple degraded versions for each image, preserving original annotations (annotations unchanged because degradation doesn't change object positions).
    Note: Rotation degradation changes coordinates and is excluded by default.
    """

    def __init__(self, seed: int = 42):
        self.degrader = ImageDegrader(seed=seed)

    def generate(self, input_image_dir: str, input_label_dir: str,
                 output_dir: str, config: Dict = None):
        """
        Generate degraded dataset.

        Args:
            input_image_dir: Clean image directory
            input_label_dir: Label directory (YOLO txt format)
            output_dir: Output directory (automatically creates images/ and labels/ subdirectories)
            config: Degradation config, generates all degradation types at all severity levels by default

        Default config:
        {
            "degradation_types": ["gaussian_noise", "darken", "uneven_illumination",
                                   "gaussian_blur", "color_bias", "low_contrast"],
            "severities": ["mild", "moderate", "severe"],
            "scenarios": ["dark_lab", "fluorescent", "phone_capture", "old_equipment", "edge_lighting"],
            "combined": {"num": 3, "num_degradations": 2, "severity": "moderate"},
            "include_clean": true
        }
        """
        if config is None:
            config = {
                "degradation_types": [
                    "gaussian_noise", "darken", "uneven_illumination",
                    "gaussian_blur", "color_bias", "low_contrast",
                ],
                "severities": ["mild", "moderate", "severe"],
                "scenarios": ["dark_lab", "fluorescent", "phone_capture",
                             "old_equipment", "edge_lighting"],
                "combined": {"num": 3, "num_degradations": 2, "severity": "moderate"},
                "include_clean": True,
            }

        # Create output directories
        out_images = os.path.join(output_dir, "images")
        out_labels = os.path.join(output_dir, "labels")
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        # Degradation method mapping
        method_map = {
            "gaussian_noise": self.degrader.add_gaussian_noise,
            "salt_pepper_noise": self.degrader.add_salt_pepper_noise,
            "darken": self.degrader.darken,
            "uneven_illumination": self.degrader.uneven_illumination,
            "gaussian_blur": self.degrader.gaussian_blur,
            "motion_blur": self.degrader.motion_blur,
            "color_bias": self.degrader.color_bias,
            "low_contrast": self.degrader.low_contrast,
        }

        # Collect image list
        image_files = sorted([f for f in os.listdir(input_image_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        metadata = []
        total_generated = 0

        for img_idx, img_file in enumerate(image_files):
            img_path = os.path.join(input_image_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + ".txt"
            label_path = os.path.join(input_label_dir, label_file)

            # Read labels (if exists)
            label_content = ""
            if os.path.exists(label_path):
                with open(label_path) as f:
                    label_content = f.read()

            # 1. Keep clean original (if config requires)
            if config.get("include_clean", True):
                clean_name = f"{base_name}_clean"
                cv2.imwrite(os.path.join(out_images, clean_name + ".jpg"), image)
                with open(os.path.join(out_labels, clean_name + ".txt"), "w") as f:
                    f.write(label_content)
                metadata.append({
                    "original": img_file,
                    "generated": clean_name + ".jpg",
                    "degradations": [],
                    "type": "clean",
                })
                total_generated += 1

            # 2. Single degradation × each severity
            for deg_type in config.get("degradation_types", []):
                if deg_type not in method_map:
                    continue
                method = method_map[deg_type]
                for severity in config.get("severities", ["moderate"]):
                    degraded, record = method(image, severity)
                    out_name = f"{base_name}_{deg_type}_{severity}"
                    cv2.imwrite(os.path.join(out_images, out_name + ".jpg"), degraded)
                    with open(os.path.join(out_labels, out_name + ".txt"), "w") as f:
                        f.write(label_content)
                    metadata.append({
                        "original": img_file,
                        "generated": out_name + ".jpg",
                        "degradations": [{"type": record.degradation_type,
                                         "params": record.params,
                                         "severity": record.severity}],
                        "type": "single",
                    })
                    total_generated += 1

            # 3. Predefined scenarios
            for scenario in config.get("scenarios", []):
                degraded, records = self.degrader.create_scenario(image, scenario)
                out_name = f"{base_name}_scenario_{scenario}"
                cv2.imwrite(os.path.join(out_images, out_name + ".jpg"), degraded)
                with open(os.path.join(out_labels, out_name + ".txt"), "w") as f:
                    f.write(label_content)
                metadata.append({
                    "original": img_file,
                    "generated": out_name + ".jpg",
                    "degradations": [{"type": r.degradation_type,
                                     "params": r.params,
                                     "severity": r.severity} for r in records],
                    "type": "scenario",
                    "scenario": scenario,
                })
                total_generated += 1

            # 4. Random combined degradation
            combined_config = config.get("combined", {})
            for combo_idx in range(combined_config.get("num", 3)):
                degraded, records = self.degrader.random_combined(
                    image,
                    num_degradations=combined_config.get("num_degradations", 2),
                    severity=combined_config.get("severity", "moderate"),
                )
                out_name = f"{base_name}_combined_{combo_idx}"
                cv2.imwrite(os.path.join(out_images, out_name + ".jpg"), degraded)
                with open(os.path.join(out_labels, out_name + ".txt"), "w") as f:
                    f.write(label_content)
                metadata.append({
                    "original": img_file,
                    "generated": out_name + ".jpg",
                    "degradations": [{"type": r.degradation_type,
                                     "params": r.params,
                                     "severity": r.severity} for r in records],
                    "type": "combined",
                })
                total_generated += 1

            if (img_idx + 1) % 100 == 0:
                print(f"Processed {img_idx + 1}/{len(image_files)} images, "
                      f"generated {total_generated} samples")

        # Save metadata
        metadata_path = os.path.join(output_dir, "degradation_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_original": len(image_files),
                "total_generated": total_generated,
                "config": config,
                "samples": metadata,
            }, f, indent=2, ensure_ascii=False)

        print(f"\nDone! Generated {total_generated} degraded images from "
              f"{len(image_files)} originals.")
        print(f"Output: {output_dir}")
        print(f"Metadata: {metadata_path}")

        return metadata
