"""
Convert camera frame + LIDAR scan + text command into RL observation vector.

Observation layout (total: 518 dims):
  [0:2]   pixel_center_norm (x, y in [-1, 1]) — target bbox center
  [2]     semantic_match    (CLIP cosine similarity, float)
  [3]     target_detected   (1.0 or 0.0)
  [4:184] lidar_compressed  (180 readings, every 2°, range-normalized)
  [184:]  text_embedding    (CLIP text embed, 334 dims after PCA — or raw 512)

When no target is detected, pixel_center_norm = [0,0] and semantic_match = 0.
"""

import numpy as np
from typing import Optional, Tuple
from .vision_pipeline import VisionPipeline


# Observation layout constants
IDX_PIXEL_X    = 0
IDX_PIXEL_Y    = 1
IDX_SEM_MATCH  = 2
IDX_DETECTED   = 3
IDX_LIDAR_START = 4
LIDAR_BINS     = 180
IDX_EMBED_START = IDX_LIDAR_START + LIDAR_BINS
EMBED_DIM      = 512
OBS_DIM        = IDX_EMBED_START + EMBED_DIM  # = 696


class VisionObsBuilder:
    """
    Translate raw sensor data into a fixed-size RL observation vector.
    Caches text embedding per command (recomputed only on command change).
    """

    def __init__(self, image_w: int = 640, image_h: int = 480, lidar_max_range: float = 3.5):
        """
        Args:
            image_w: Camera image width in pixels
            image_h: Camera image height in pixels
            lidar_max_range: LIDAR maximum range in meters (for normalization)
        """
        self.image_w = image_w
        self.image_h = image_h
        self.lidar_max_range = lidar_max_range

        self.pipeline = VisionPipeline()

        self._cached_command: Optional[str] = None
        self._cached_text_embed: Optional[np.ndarray] = None

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    def build(
        self,
        image: np.ndarray,
        command: str,
        lidar_ranges: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Build observation vector from sensor inputs.

        Args:
            image: RGB image (H, W, 3), uint8
            command: Natural language navigation target
            lidar_ranges: LIDAR range array (360 readings at 1°/step), float32

        Returns:
            (obs, info)
            obs: float32 array of shape (OBS_DIM,)
            info: {
                "target_box": [...] or None,
                "target_class": str or None,
                "semantic_match": float,
                "detected": bool,
            }
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # ── Text embedding (cached) ──────────────────────────────────────
        text_embed = self._get_text_embed(command)
        obs[IDX_EMBED_START:] = text_embed

        # ── Vision pipeline ───────────────────────────────────────────────
        vision_result = self.pipeline.process(image, command)
        detected = vision_result["target_box"] is not None

        obs[IDX_DETECTED] = 1.0 if detected else 0.0
        obs[IDX_SEM_MATCH] = float(vision_result["semantic_match"])

        if detected:
            cx, cy = vision_result["target_center"]
            # Normalize to [-1, 1]
            obs[IDX_PIXEL_X] = (cx / self.image_w) * 2.0 - 1.0
            obs[IDX_PIXEL_Y] = (cy / self.image_h) * 2.0 - 1.0

        # ── LIDAR (compress 360 → 180 bins) ──────────────────────────────
        if lidar_ranges is not None:
            obs[IDX_LIDAR_START:IDX_EMBED_START] = self._compress_lidar(lidar_ranges)

        info = {
            "target_box":     vision_result["target_box"],
            "target_class":   vision_result["target_class"],
            "semantic_match": vision_result["semantic_match"],
            "detected":       detected,
        }

        return obs, info

    def _get_text_embed(self, command: str) -> np.ndarray:
        """Return cached CLIP text embed, recompute only on command change."""
        if command != self._cached_command:
            self._cached_text_embed = self.pipeline.encoder.encode_text(command)
            self._cached_command = command
        return self._cached_text_embed

    def _compress_lidar(self, ranges: np.ndarray) -> np.ndarray:
        """
        Compress 360-reading LIDAR to 180 bins by taking min in each 2° window.
        Normalize by max range. Replace inf/nan with 1.0 (max).
        """
        ranges = np.asarray(ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, self.lidar_max_range)
        ranges = np.clip(ranges, 0.0, self.lidar_max_range)

        # Reshape to (180, 2), take min per bin
        n = (len(ranges) // LIDAR_BINS) * LIDAR_BINS
        compressed = ranges[:n].reshape(LIDAR_BINS, -1).min(axis=1)
        compressed /= self.lidar_max_range  # Normalize to [0, 1]

        return compressed
