"""
Top-down 2D renderer for NavEnv episodes.
Produces RGB frames (H×W×3 uint8) suitable for GIF/MP4 encoding.

Frame layout:
  - Dark grid background (10×10m room)
  - Gray filled circles: obstacles
  - Green star: target
  - Blue circle + heading arrow: robot
  - Red polyline: trajectory trail
  - Top-left HUD: step / distance / reward / success flag
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe in headless/subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from typing import List, Optional, Tuple
import io
from PIL import Image


# Canvas constants
CANVAS_PX   = 512       # Square frame size in pixels
DPI         = 100
FIG_SIZE_IN = CANVAS_PX / DPI
ROOM_SIZE   = 10.0      # Must match nav_env.ROOM_SIZE
HALF        = ROOM_SIZE / 2


class NavEnvRenderer:
    """Render NavEnv state as RGB image frames."""

    def __init__(self, canvas_px: int = CANVAS_PX, trail_len: int = 80):
        """
        Args:
            canvas_px: Output frame size (square)
            trail_len: Number of past positions to draw as trajectory trail
        """
        self.canvas_px = canvas_px
        self.trail_len = trail_len
        self._trail: List[Tuple[float, float]] = []

        # Build figure once, reuse every frame
        self._fig, self._ax = plt.subplots(figsize=(FIG_SIZE_IN, FIG_SIZE_IN), dpi=DPI)
        self._setup_axes()

    def reset(self):
        """Clear trail at episode start."""
        self._trail.clear()

    def render_frame(
        self,
        robot_pos: np.ndarray,
        robot_theta: float,
        target_pos: np.ndarray,
        obstacles: List[np.ndarray],
        step: int,
        distance: float,
        ep_reward: float,
        success: bool = False,
        command: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render one frame.

        Args:
            robot_pos:   (x, y) robot position in meters
            robot_theta: Robot heading in radians
            target_pos:  (x, y) target position in meters
            obstacles:   List of (x, y, radius) arrays
            step:        Current step count
            distance:    Distance to target (meters)
            ep_reward:   Cumulative episode reward
            success:     True when goal was reached this frame
            command:     Optional text command to display as subtitle

        Returns:
            RGB frame as uint8 array (canvas_px, canvas_px, 3)
        """
        ax = self._ax
        ax.clear()
        self._setup_axes()

        # ── Trail ─────────────────────────────────────────────────────────
        self._trail.append((float(robot_pos[0]), float(robot_pos[1])))
        if len(self._trail) > self.trail_len:
            self._trail.pop(0)

        if len(self._trail) > 1:
            xs, ys = zip(*self._trail)
            alphas = np.linspace(0.1, 0.7, len(self._trail))
            for i in range(len(self._trail) - 1):
                ax.plot(
                    [xs[i], xs[i+1]], [ys[i], ys[i+1]],
                    color="#e74c3c", alpha=float(alphas[i]), linewidth=1.5,
                )

        # ── Obstacles ──────────────────────────────────────────────────────
        for obs in obstacles:
            circle = plt.Circle(
                (obs[0], obs[1]), obs[2],
                color="#7f8c8d", alpha=0.75, zorder=2,
            )
            ax.add_patch(circle)

        # ── Target ────────────────────────────────────────────────────────
        target_color = "#2ecc71" if not success else "#f1c40f"
        ax.plot(
            target_pos[0], target_pos[1],
            marker="*", markersize=18,
            color=target_color, zorder=4,
            markeredgecolor="white", markeredgewidth=0.5,
        )
        # Goal radius ring
        goal_ring = plt.Circle(
            (target_pos[0], target_pos[1]), 0.5,
            fill=False, color=target_color, alpha=0.35, linestyle="--", linewidth=1,
        )
        ax.add_patch(goal_ring)

        # ── Robot ─────────────────────────────────────────────────────────
        robot_color = "#3498db" if not success else "#f39c12"
        robot_circle = plt.Circle(
            (robot_pos[0], robot_pos[1]), 0.18,
            color=robot_color, zorder=5,
        )
        ax.add_patch(robot_circle)

        # Heading arrow
        arrow_len = 0.45
        dx = arrow_len * np.cos(robot_theta)
        dy = arrow_len * np.sin(robot_theta)
        ax.annotate(
            "", xy=(robot_pos[0] + dx, robot_pos[1] + dy),
            xytext=(robot_pos[0], robot_pos[1]),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.8),
            zorder=6,
        )

        # ── HUD ───────────────────────────────────────────────────────────
        status = "REACHED" if success else f"dist={distance:.2f}m"
        hud = (
            f"step={step:4d}   {status}\n"
            f"reward={ep_reward:+6.1f}"
        )
        ax.text(
            -HALF + 0.2, HALF - 0.3, hud,
            fontsize=7, color="white", family="monospace",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.55),
            zorder=7,
        )

        # ── Command subtitle ───────────────────────────────────────────────
        if command:
            ax.set_title(f'"{command}"', fontsize=8, color="#ecf0f1", pad=4)

        return self._fig_to_array()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _setup_axes(self):
        ax = self._ax
        ax.set_xlim(-HALF, HALF)
        ax.set_ylim(-HALF, HALF)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        self._fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#555", labelsize=6)
        ax.spines[:].set_color("#333")

        # Light grid
        for v in np.arange(-HALF, HALF + 1, 1):
            ax.axhline(v, color="#2c2c4a", linewidth=0.4)
            ax.axvline(v, color="#2c2c4a", linewidth=0.4)

        # Room boundary
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("#4a4a6a")

    def _fig_to_array(self) -> np.ndarray:
        """Convert matplotlib figure to RGB uint8 array."""
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = np.array(Image.open(buf).convert("RGB"))
        buf.close()
        return img.astype(np.uint8)

    def close(self):
        plt.close(self._fig)
