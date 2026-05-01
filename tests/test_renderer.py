"""Test NavEnvRenderer frame generation."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "langnav/ros2_ws/src/langnav_robot"))

from langnav_rl.renderer import NavEnvRenderer


@pytest.fixture(scope="module")
def renderer():
    r = NavEnvRenderer(canvas_px=256, trail_len=20)
    yield r
    r.close()


@pytest.fixture
def obstacles():
    return [
        np.array([2.0, 1.5, 0.3]),
        np.array([-1.5, -2.0, 0.4]),
    ]


class TestNavEnvRendererFrame:
    def test_frame_shape(self, renderer, obstacles):
        """render_frame returns (H, W, 3) uint8."""
        frame = renderer.render_frame(
            robot_pos   = np.array([0.0, 0.0]),
            robot_theta = 0.0,
            target_pos  = np.array([2.0, 2.0]),
            obstacles   = obstacles,
            step        = 1,
            distance    = 2.83,
            ep_reward   = 0.0,
        )
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8

    def test_frame_nonzero(self, renderer, obstacles):
        """Frame should not be entirely black."""
        frame = renderer.render_frame(
            robot_pos   = np.array([1.0, 1.0]),
            robot_theta = 0.785,
            target_pos  = np.array([-2.0, -2.0]),
            obstacles   = obstacles,
            step        = 50,
            distance    = 4.24,
            ep_reward   = -12.5,
        )
        assert frame.max() > 10

    def test_success_flag_accepted(self, renderer):
        """success=True renders without error."""
        renderer.render_frame(
            robot_pos   = np.array([0.1, 0.0]),
            robot_theta = 0.0,
            target_pos  = np.array([0.0, 0.0]),
            obstacles   = [],
            step        = 100,
            distance    = 0.1,
            ep_reward   = 9.5,
            success     = True,
        )

    def test_command_subtitle(self, renderer):
        """command kwarg renders without error."""
        renderer.render_frame(
            robot_pos   = np.array([0.0, 0.0]),
            robot_theta = 0.0,
            target_pos  = np.array([3.0, 0.0]),
            obstacles   = [],
            step        = 1,
            distance    = 3.0,
            ep_reward   = 0.0,
            command     = "go to the red box",
        )

    def test_no_obstacles(self, renderer):
        """Empty obstacle list renders without error."""
        renderer.render_frame(
            robot_pos   = np.array([0.0, 0.0]),
            robot_theta = 0.0,
            target_pos  = np.array([1.0, 1.0]),
            obstacles   = [],
            step        = 1,
            distance    = 1.41,
            ep_reward   = 0.0,
        )


class TestNavEnvRendererTrail:
    def test_reset_clears_trail(self, renderer, obstacles):
        """Trail cleared on reset()."""
        for i in range(10):
            renderer.render_frame(
                robot_pos   = np.array([float(i) * 0.1, 0.0]),
                robot_theta = 0.0,
                target_pos  = np.array([3.0, 0.0]),
                obstacles   = [],
                step        = i,
                distance    = 3.0 - i * 0.1,
                ep_reward   = float(i),
            )
        assert len(renderer._trail) > 0
        renderer.reset()
        assert len(renderer._trail) == 0

    def test_trail_capped_at_trail_len(self, renderer, obstacles):
        """Trail doesn't grow beyond trail_len."""
        renderer.reset()
        for i in range(50):
            renderer.render_frame(
                robot_pos   = np.array([float(i) * 0.05, 0.0]),
                robot_theta = 0.0,
                target_pos  = np.array([3.0, 0.0]),
                obstacles   = [],
                step        = i,
                distance    = 3.0,
                ep_reward   = 0.0,
            )
        assert len(renderer._trail) <= renderer.trail_len


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
