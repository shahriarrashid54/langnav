"""Test vision pipeline integration."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "langnav/ros2_ws/src/langnav_robot"))

from langnav_vision import VisionPipeline, YOLODetector, CLIPEncoder


class TestYOLODetector:
    """Test YOLO object detection."""

    def test_detector_init(self):
        """Detector initializes without error."""
        detector = YOLODetector(model_name="yolov11n.pt", conf_threshold=0.5)
        assert detector is not None
        assert detector.conf_threshold == 0.5

    def test_detector_detect(self):
        """Detector returns structured detections."""
        detector = YOLODetector()
        # Dummy 640x480 RGB image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image)

        assert "boxes" in detections
        assert "classes" in detections
        assert "confs" in detections
        assert isinstance(detections["boxes"], list)


class TestCLIPEncoder:
    """Test CLIP vision-language encoding."""

    def test_encoder_init(self):
        """Encoder initializes without error."""
        encoder = CLIPEncoder(model_name="ViT-B/32", device="cpu")
        assert encoder is not None

    def test_encode_text(self):
        """Text encoding returns 512-D embedding."""
        encoder = CLIPEncoder(device="cpu")
        embedding = encoder.encode_text("go to the red chair")

        assert embedding.ndim == 1
        assert embedding.shape[0] in [512, 768]  # ViT-B/32 or ViT-L/14


class TestVisionPipeline:
    """Test end-to-end vision pipeline."""

    def test_pipeline_init(self):
        """Pipeline initializes."""
        pipeline = VisionPipeline()
        assert pipeline.detector is not None
        assert pipeline.encoder is not None

    def test_pipeline_process(self):
        """Pipeline processes image + command."""
        pipeline = VisionPipeline()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        command = "find the red box"

        result = pipeline.process(image, command)

        assert "target_box" in result
        assert "target_class" in result
        assert "confidence" in result
        assert "semantic_match" in result
        assert "target_center" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
