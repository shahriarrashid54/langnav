"""CLIP vision-language encoder for semantic understanding."""

import clip
import torch
import numpy as np
from typing import List, Dict, Tuple


class CLIPEncoder:
    """Map text commands + images to semantic embeddings."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Initialize CLIP model.

        Args:
            model_name: CLIP model variant
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode natural language command to embedding.

        Args:
            text: Natural language instruction (e.g., "go to the red chair")

        Returns:
            Text embedding (512 or 768 dim depending on model)
        """
        with torch.no_grad():
            tokens = clip.tokenize(text).to(self.device)
            embedding = self.model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0]

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image to embedding.

        Args:
            image: RGB image (H, W, 3), uint8

        Returns:
            Image embedding
        """
        with torch.no_grad():
            image_tensor = self.preprocess(
                self._bgr_to_pil(image)
            ).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0]

    def encode_crop(self, image: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Encode image crop (bounding box region).

        Args:
            image: Full RGB image
            box: [x1, y1, x2, y2] bounding box

        Returns:
            Crop embedding
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = image[y1:y2, x1:x2]
        return self.encode_image(crop)

    def match_text_to_objects(
        self, text: str, crops: List[np.ndarray], top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Find which object crops best match the text command.

        Args:
            text: Natural language instruction
            crops: List of image crops
            top_k: Return top K matches

        Returns:
            [(crop_idx, similarity_score), ...]
        """
        text_embedding = self.encode_text(text)
        similarities = []

        for i, crop in enumerate(crops):
            crop_embedding = self.encode_image(crop)
            sim = np.dot(text_embedding, crop_embedding)
            similarities.append((i, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    @staticmethod
    def _bgr_to_pil(image: np.ndarray):
        """Convert BGR numpy image to PIL Image."""
        from PIL import Image
        rgb = image[..., ::-1]
        return Image.fromarray(rgb)
