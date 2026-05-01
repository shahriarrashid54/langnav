"""End-to-end vision pipeline: YOLO detection + CLIP matching."""

import numpy as np
from typing import Dict, Tuple, Optional
from .yolo_detector import YOLODetector
from .clip_encoder import CLIPEncoder


class VisionPipeline:
    """Map natural language commands to detected objects via YOLO + CLIP."""

    def __init__(self, yolo_model: str = "yolov11n.pt", clip_model: str = "ViT-B/32"):
        """
        Initialize vision pipeline.

        Args:
            yolo_model: YOLOv11 model variant
            clip_model: CLIP model variant
        """
        self.detector = YOLODetector(model_name=yolo_model, conf_threshold=0.5)
        self.encoder = CLIPEncoder(model_name=clip_model)

    def process(self, image: np.ndarray, command: str) -> Dict:
        """
        Process image + command to find target object.

        Args:
            image: RGB image (H, W, 3)
            command: Natural language command (e.g., "go to the red chair")

        Returns:
            {
                "target_box": [x1, y1, x2, y2],
                "target_class": "chair",
                "confidence": 0.89,
                "semantic_match": 0.92,
                "target_center": [x, y],
                "all_detections": [...]
            }
        """
        # Detect all objects
        detections = self.detector.detect(image)

        if not detections["boxes"]:
            return {
                "target_box": None,
                "target_class": None,
                "confidence": 0.0,
                "semantic_match": 0.0,
                "target_center": None,
                "all_detections": [],
            }

        # Extract crops for each detection
        crops = [
            self._extract_crop(image, box)
            for box in detections["boxes"]
        ]

        # Find best semantic match to command
        matches = self.encoder.match_text_to_objects(command, crops, top_k=1)

        if not matches:
            return {
                "target_box": None,
                "target_class": None,
                "confidence": 0.0,
                "semantic_match": 0.0,
                "target_center": None,
                "all_detections": detections,
            }

        best_idx, semantic_sim = matches[0]
        target_box = detections["boxes"][best_idx]
        target_class = detections["classes"][best_idx]
        detection_conf = detections["confs"][best_idx]
        target_center = self._box_to_center(target_box)

        return {
            "target_box": target_box,
            "target_class": target_class,
            "confidence": detection_conf,
            "semantic_match": semantic_sim,
            "target_center": target_center,
            "all_detections": detections,
        }

    @staticmethod
    def _extract_crop(image: np.ndarray, box: Tuple[float, ...]) -> np.ndarray:
        """Extract image region from bounding box."""
        x1, y1, x2, y2 = [int(v) for v in box]
        return image[y1:y2, x1:x2]

    @staticmethod
    def _box_to_center(box: Tuple[float, ...]) -> Tuple[float, float]:
        """Convert bounding box to center coordinates."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
