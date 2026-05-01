"""
Faster R-CNN Detection Tool - Object detection using torchvision.
"""

import numpy as np
from typing import Dict, List

from tools.base import BaseTool, ToolResult
from tools.registry import register_tool

# COCO class names (91 categories, index 0 = __background__)
COCO_CLASS_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


@register_tool
class FasterRCNNDetectTool(BaseTool):
    """Faster R-CNN object detection tool using torchvision."""

    name = "fasterrcnn_detect"
    description = "Detect objects in an image using Faster R-CNN from torchvision."
    parameters = {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "default": "fasterrcnn_resnet50_fpn_v2",
                "description": "Model architecture name from torchvision.models.detection.",
            },
            "conf_threshold": {
                "type": "number",
                "default": 0.5,
                "description": "Confidence threshold for detections.",
            },
            "device": {
                "type": "string",
                "default": "cuda",
                "description": "Device to run inference on (cuda/cpu).",
            },
            "model_path": {
                "type": "string",
                "default": "",
                "description": "Path to custom model weights file.",
            },
            "num_classes": {
                "type": "integer",
                "default": 91,
                "description": "Number of classes (91 for COCO pretrained).",
            },
        },
        "required": [],
    }

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self._model = None
        self._loaded_path: str = ""
        self._device = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self, model_path: str, model_name: str, device: str, num_classes: int):
        """Load Faster R-CNN model on first call or when model changes."""
        load_key = model_path or model_name
        if self._model is not None and self._loaded_path == load_key:
            return

        import torch
        import torchvision.models.detection as det_models
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        # Resolve device
        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        # Get the model constructor
        model_fn = getattr(det_models, model_name, None)
        if model_fn is None:
            raise ValueError(f"Unknown model '{model_name}'.")

        # Load custom weights (prefer local weights, don't rely on pretrained)
        import os
        if model_path and os.path.isfile(model_path):
            # Directly use local weights, don't load pretrained
            self._model = model_fn(weights=None)
            # Always replace detection head with correct number of classes
            in_features = self._model.roi_heads.box_predictor.cls_score.in_features
            self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            state_dict = torch.load(model_path, map_location=self._device, weights_only=False)
            self._model.load_state_dict(state_dict)
        elif num_classes == 91:
            self._model = model_fn(weights="DEFAULT")
        else:
            self._model = model_fn(weights=None, num_classes=num_classes)

        self._model.to(self._device)
        self._model.eval()
        self._loaded_path = load_key

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(image: np.ndarray, device, imgsz: int = 640):
        """Convert HWC uint8 numpy image to a normalised float32 tensor."""
        import torch
        import cv2

        # Resize to model input size
        img = cv2.resize(image, (imgsz, imgsz))

        # Ensure RGB float32 in [0, 1]
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        # BGR -> RGB
        img = img[:, :, ::-1].copy()
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW tensor
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).to(device)
        return tensor

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_results(outputs: Dict, conf_threshold: float, num_classes: int) -> List[Dict]:
        """Parse torchvision detection outputs into a unified detection list."""
        detections: List[Dict] = []
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        for i in range(len(scores)):
            conf = float(scores[i])
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(float, boxes[i])
            cls = int(labels[i])

            if num_classes == 91 and cls < len(COCO_CLASS_NAMES):
                class_name = COCO_CLASS_NAMES[cls]
            else:
                # Custom trained model: class 1-N corresponds to actual classes 0-(N-1)
                # Faster R-CNN's class 0 is background
                cls_adjusted = max(cls - 1, 0)
                class_name = f"class_{cls_adjusted}"
                cls = cls_adjusted

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls,
                "class_name": class_name,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "width": x2 - x1,
                "height": y2 - y1,
            })

        return detections

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def call(self, image: np.ndarray, **params) -> ToolResult:
        model_path: str = params.get("model_path", "")
        model_name: str = params.get("model_name", "fasterrcnn_resnet50_fpn_v2")
        conf_threshold: float = params.get("conf_threshold", 0.5)
        device: str = params.get("device", "cuda")
        num_classes: int = params.get("num_classes", 91)

        try:
            self._load_model(model_path, model_name, device, num_classes)
        except ImportError:
            return ToolResult(
                image=image,
                detections=[],
                metadata={"model": "fasterrcnn", "model_name": model_name},
                success=False,
                error="torch/torchvision is not installed. Install with: pip install torch torchvision",
            )
        except Exception as exc:
            return ToolResult(
                image=image,
                detections=[],
                metadata={"model": "fasterrcnn", "model_name": model_name},
                success=False,
                error=f"Failed to load Faster R-CNN model: {exc}",
            )

        try:
            import torch

            orig_h, orig_w = image.shape[:2]
            imgsz = 640
            tensor = self._preprocess(image, self._device, imgsz)

            with torch.no_grad():
                outputs = self._model([tensor])[0]

            detections = self._parse_results(outputs, conf_threshold, num_classes)

            # Scale detection boxes from imgsz back to original image size
            scale_x = orig_w / imgsz
            scale_y = orig_h / imgsz
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                d["bbox"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                d["center"] = [(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2]
                d["width"] = d["bbox"][2] - d["bbox"][0]
                d["height"] = d["bbox"][3] - d["bbox"][1]

            return ToolResult(
                image=image,
                detections=detections,
                metadata={
                    "model": "fasterrcnn",
                    "model_name": model_name,
                    "conf_threshold": conf_threshold,
                    "device": str(self._device),
                    "num_classes": num_classes,
                    "num_detections": len(detections),
                },
                success=True,
            )
        except Exception as exc:
            return ToolResult(
                image=image,
                detections=[],
                metadata={"model": "fasterrcnn", "model_name": model_name},
                success=False,
                error=f"Faster R-CNN detection failed: {exc}",
            )
