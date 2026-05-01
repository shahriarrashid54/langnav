"""YOLOv11 real-time object detection."""

from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict


class YOLODetector:
    """Detect objects in RGB images using YOLOv11."""

    def __init__(self, model_name: str = "yolov11n.pt", conf_threshold: float = 0.5):
        """
        Initialize YOLOv11 detector.

        Args:
            model_name: Model variant (yolov11n/s/m/l/x)
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect objects in image.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            {
                "boxes": [[x1, y1, x2, y2], ...],
                "classes": ["person", "chair", ...],
                "confs": [0.95, 0.87, ...],
                "class_ids": [0, 56, ...]
            }
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        r = results[0]

        detections = {
            "boxes": r.boxes.xyxy.cpu().numpy().tolist(),
            "classes": [self.model.names[int(cid)] for cid in r.boxes.cls],
            "confs": r.boxes.conf.cpu().numpy().tolist(),
            "class_ids": r.boxes.cls.cpu().numpy().tolist(),
        }
        return detections

    def get_class_names(self) -> List[str]:
        """Get all class names YOLO can detect."""
        return list(self.model.names.values())
