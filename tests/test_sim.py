"""Test simulation world generation (no ROS2 required)."""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "langnav/ros2_ws/src/langnav_robot"))

from langnav_sim.worlds.world_generator import WorldGenerator, OBJECT_PALETTE, ObjectSpec


class TestWorldGenerator:
    """Test world generation helpers (static methods, no ROS2)."""

    def test_sample_positions_count(self):
        """Returns requested number of positions."""
        positions = WorldGenerator._sample_positions(5, (-4.0, 4.0), min_dist=1.0)
        assert len(positions) == 5

    def test_sample_positions_min_dist_from_origin(self):
        """No position closer than min_dist to origin."""
        positions = WorldGenerator._sample_positions(10, (-4.0, 4.0), min_dist=1.5)
        for x, y in positions:
            dist = math.sqrt(x**2 + y**2)
            assert dist >= 1.5, f"Position ({x:.2f}, {y:.2f}) too close to origin"

    def test_sample_positions_min_spacing(self):
        """Objects maintain minimum spacing between each other."""
        min_spacing = 1.2
        positions = WorldGenerator._sample_positions(
            10, (-4.0, 4.0), min_dist=1.0, min_spacing=min_spacing
        )
        for i, (x1, y1) in enumerate(positions):
            for j, (x2, y2) in enumerate(positions):
                if i == j:
                    continue
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                assert dist >= min_spacing * 0.99, (
                    f"Objects {i} and {j} too close: {dist:.2f} < {min_spacing}"
                )

    def test_sample_positions_within_bounds(self):
        """All positions within room bounds."""
        bounds = (-4.0, 4.0)
        positions = WorldGenerator._sample_positions(5, bounds, min_dist=0.5)
        for x, y in positions:
            assert bounds[0] <= x <= bounds[1]
            assert bounds[0] <= y <= bounds[1]

    def test_build_sdf_contains_name(self):
        """Generated SDF XML references object name."""
        spec = OBJECT_PALETTE[0]
        sdf = WorldGenerator._build_sdf("test_obj", spec, (1.0, 2.0))
        assert "test_obj" in sdf
        assert "<sdf" in sdf

    def test_build_sdf_color_embedded(self):
        """SDF contains correct color values."""
        spec = ObjectSpec("test_obj", (0.8, 0.1, 0.1, 1.0), (0.5, 0.5, 0.5))
        sdf = WorldGenerator._build_sdf("test_obj", spec, (0.0, 0.0))
        assert "0.8 0.1 0.1 1.0" in sdf

    def test_object_palette_nonempty(self):
        """Palette has at least one object."""
        assert len(OBJECT_PALETTE) > 0
        for spec in OBJECT_PALETTE:
            assert len(spec.color) == 4
            assert len(spec.size) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
