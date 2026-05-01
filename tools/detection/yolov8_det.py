"""
YOLOv8 Detection Tool - Object detection using Ultralytics YOLOv8.
"""

import numpy as np
from typing import Dict, List

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool


@register_tool
class YOLOv8DetectTool(BaseTool):
    """YOLOv8 object detection tool using the Ultralytics library."""

    name = "yolov8_detect"
    description = "Detect objects in an image using YOLOv8."
    parameters = {
        "type": "object",
        "properties": {
            "model_path": {
                "type": "string",
                "default": "yolov8n.pt",
                "description": "Path to YOLOv8 model weights.",
            },
            "conf_threshold": {
                "type": "number",
                "default": 0.5,
                "description": "Confidence threshold for detections.",
            },
            "iou_threshold": {
                "type": "number",
                "default": 0.45,
                "description": "IoU threshold for NMS.",
            },
            "device": {
                "type": "string",
                "default": "cpu",
                "description": "Device to run inference on (cuda/cpu).",
            },
        },
        "required": [],
    }

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self._model = None
        self._loaded_model_path: str = ""

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self, model_path: str, device: str):
        """Load YOLOv8 model on first call or when model_path changes."""
        if self._model is not None and self._loaded_model_path == model_path:
            return

        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._model.to(device)
        self._loaded_model_path = model_path

    # ------------------------------------------------------------------
    # Result parsing (follows MicroAgent detector.py logic)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_results(results) -> List[Dict]:
        """Parse ultralytics Results objects into a unified detection list."""
        detections: List[Dict] = []
        det_id = 0

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(float, xyxy)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names.get(cls, f"class_{cls}")

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": class_name,
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    "width": x2 - x1,
                    "height": y2 - y1,
                })
                det_id += 1

        return detections

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def call(self, image: np.ndarray, **params) -> ToolResult:
        model_path: str = params.get("model_path", "yolov8n.pt")
        conf_threshold: float = params.get("conf_threshold", 0.5)
        iou_threshold: float = params.get("iou_threshold", 0.45)
        device: str = params.get("device", "cuda")
        batch_images: List[np.ndarray] = params.get("batch_images", None)

        try:
            self._load_model(model_path, device)
        except ImportError:
            return ToolResult(
                image=image,
                detections=[],
                metadata={"model": "yolov8", "model_path": model_path},
                success=False,
                error="ultralytics is not installed. Install with: pip install ultralytics",
            )
        except Exception as exc:
            return ToolResult(
                image=image,
                detections=[],
                metadata={"model": "yolov8", "model_path": model_path},
                success=False,
                error=f"Failed to load YOLOv8 model: {exc}",
            )

        try:
            # Batch inference if multiple images provided
            if batch_images and len(batch_images) > 1:
                results = self._model(
                    batch_images,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                # Parse results for each image
                all_detections = []
                for result in results:
                    img_detections = self._parse_results([result])
                    all_detections.append(img_detections)
                num_detections = sum(len(d) for d in all_detections)
                return ToolResult(
                    image=image,
                    detections=all_detections,
                    metadata={
                        "model": "yolov8",
                        "model_path": model_path,
                        "conf_threshold": conf_threshold,
                        "iou_threshold": iou_threshold,
                        "device": device,
                        "batch_size": len(batch_images),
                        "num_detections": num_detections,
                    },
                    success=True,
                )
            else:
                # Single image inference
                results = self._model(
                    image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                detections = self._parse_results(results)
                return ToolResult(
                    image=image,
                    detections=detections,
                    metadata={
                        "model": "yolov8",
                        "model_path": model_path,
                        "conf_threshold": conf_threshold,
                        "iou_threshold": iou_threshold,
                        "device": device,
                        "num_detections": len(detections),
                    },
                    success=True,
                )
        except Exception as exc:
            return ToolResult(
                image=image,
                detections=[],
                metadata={"model": "yolov8", "model_path": model_path},
                success=False,
                error=f"YOLOv8 detection failed: {exc}",
            )
