"""Test VisionObsBuilder — sensor fusion layer for RL obs construction."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "langnav/ros2_ws/src/langnav_robot"))

from langnav_vision import VisionObsBuilder, OBS_DIM
from langnav_vision.vision_obs_builder import (
    IDX_PIXEL_X, IDX_PIXEL_Y, IDX_SEM_MATCH, IDX_DETECTED,
    IDX_LIDAR_START, IDX_EMBED_START, LIDAR_BINS,
)


@pytest.fixture
def builder():
    return VisionObsBuilder(image_w=640, image_h=480)


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_lidar():
    # 360-reading LIDAR, range 0.5–3.0m
    ranges = np.random.uniform(0.5, 3.0, 360).astype(np.float32)
    ranges[10:20] = np.inf  # Simulate missing readings
    return ranges


class TestVisionObsBuilderShape:
    def test_obs_dim_constant(self):
        """OBS_DIM matches expected size."""
        expected = IDX_EMBED_START + 512
        assert OBS_DIM == expected

    def test_build_returns_correct_shape(self, builder, dummy_image):
        obs, _ = builder.build(dummy_image, "find the red box")
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_build_with_lidar(self, builder, dummy_image, dummy_lidar):
        obs, _ = builder.build(dummy_image, "find the red box", dummy_lidar)
        assert obs.shape == (OBS_DIM,)

    def test_build_without_lidar(self, builder, dummy_image):
        obs, _ = builder.build(dummy_image, "find the red box", lidar_ranges=None)
        # LIDAR slice should be all zeros
        lidar_slice = obs[IDX_LIDAR_START:IDX_EMBED_START]
        assert np.all(lidar_slice == 0.0)


class TestVisionObsBuilderValues:
    def test_pixel_coords_in_range(self, builder, dummy_image):
        obs, info = builder.build(dummy_image, "find the red box")
        if info["detected"]:
            assert -1.0 <= obs[IDX_PIXEL_X] <= 1.0
            assert -1.0 <= obs[IDX_PIXEL_Y] <= 1.0

    def test_semantic_match_in_range(self, builder, dummy_image):
        obs, _ = builder.build(dummy_image, "find the red box")
        assert -1.0 <= obs[IDX_SEM_MATCH] <= 1.0

    def test_detected_flag_binary(self, builder, dummy_image):
        obs, info = builder.build(dummy_image, "find the red box")
        assert obs[IDX_DETECTED] in (0.0, 1.0)
        assert obs[IDX_DETECTED] == (1.0 if info["detected"] else 0.0)

    def test_text_embedding_nonzero(self, builder, dummy_image):
        obs, _ = builder.build(dummy_image, "go to the blue table")
        embed = obs[IDX_EMBED_START:]
        assert not np.all(embed == 0.0)

    def test_lidar_normalized(self, builder, dummy_image, dummy_lidar):
        obs, _ = builder.build(dummy_image, "go to the chair", dummy_lidar)
        lidar_slice = obs[IDX_LIDAR_START:IDX_EMBED_START]
        assert np.all(lidar_slice >= 0.0)
        assert np.all(lidar_slice <= 1.0)

    def test_lidar_inf_handled(self, builder, dummy_image):
        """Inf LIDAR readings should not propagate to obs."""
        all_inf = np.full(360, np.inf, dtype=np.float32)
        obs, _ = builder.build(dummy_image, "navigate", all_inf)
        lidar_slice = obs[IDX_LIDAR_START:IDX_EMBED_START]
        assert np.all(np.isfinite(lidar_slice))
        assert np.all(lidar_slice == 1.0)  # All at max range after normalization


class TestVisionObsBuilderCache:
    def test_text_embed_cached(self, builder, dummy_image):
        """Same command reuses cached embedding (same array pointer)."""
        cmd = "navigate to the red box"
        builder.build(dummy_image, cmd)
        embed1 = builder._cached_text_embed.copy()

        builder.build(dummy_image, cmd)  # Should hit cache
        embed2 = builder._cached_text_embed.copy()

        np.testing.assert_array_equal(embed1, embed2)

    def test_text_embed_updates_on_command_change(self, builder, dummy_image):
        """Different command produces different embedding."""
        builder.build(dummy_image, "go to the red box")
        embed1 = builder._cached_text_embed.copy()

        builder.build(dummy_image, "find the yellow cylinder")
        embed2 = builder._cached_text_embed.copy()

        assert not np.allclose(embed1, embed2, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
