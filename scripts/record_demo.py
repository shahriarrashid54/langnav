"""
Record GIF/MP4 demos of trained PPO navigation policy.

Outputs:
  <output_dir>/demo_best.gif      — best-success episode (GitHub README)
  <output_dir>/demo_best.mp4      — same episode as video
  <output_dir>/demo_compare.gif   — random vs trained side-by-side
  <output_dir>/eval_results.json  — full eval metrics table
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent / "langnav/ros2_ws/src/langnav_robot"))

from langnav_rl import NavEnv
from langnav_rl.renderer import NavEnvRenderer
from stable_baselines3 import PPO


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    env: NavEnv,
    model: Optional[PPO],
    renderer: NavEnvRenderer,
    command: str = "navigate to the target",
    max_steps: int = 500,
    deterministic: bool = True,
) -> dict:
    """
    Run one episode, capturing rendered frames.

    Args:
        env:          NavEnv instance
        model:        Trained PPO model (None = random policy)
        renderer:     Frame renderer
        command:      Text command for HUD subtitle
        max_steps:    Episode step cap
        deterministic: Use deterministic PPO actions

    Returns:
        {
            "frames":     list of RGB arrays,
            "reward":     total episode reward,
            "success":    bool,
            "distance":   final distance to target,
            "steps":      step count,
        }
    """
    obs, _ = env.reset()
    renderer.reset()

    frames = []
    ep_reward = 0.0
    success = False
    distance = float(np.linalg.norm(env.robot_pos - env.target_pos))

    for step in range(max_steps):
        # Get action
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        distance = info["distance"]
        success = info.get("success", False)

        frame = renderer.render_frame(
            robot_pos   = env.robot_pos,
            robot_theta = env.robot_theta,
            target_pos  = env.target_pos,
            obstacles   = env.obstacles,
            step        = step + 1,
            distance    = distance,
            ep_reward   = ep_reward,
            success     = success,
            command     = command,
        )
        frames.append(frame)

        if terminated or truncated:
            # Hold final frame for 1 second
            hold_frames = 10 if success else 5
            frames.extend([frames[-1]] * hold_frames)
            break

    return {
        "frames":   frames,
        "reward":   ep_reward,
        "success":  success,
        "distance": distance,
        "steps":    step + 1,
    }


# ── Side-by-side comparator ───────────────────────────────────────────────────

def make_comparison_gif(
    random_frames: List[np.ndarray],
    trained_frames: List[np.ndarray],
    output_path: str,
    fps: int = 15,
    separator_px: int = 4,
):
    """
    Stitch random and trained frames side-by-side into one GIF.
    Pads shorter episode with its last frame.
    """
    n = max(len(random_frames), len(trained_frames))

    def pad(frames, n):
        if len(frames) < n:
            frames = frames + [frames[-1]] * (n - len(frames))
        return frames[:n]

    r_frames = pad(random_frames, n)
    t_frames = pad(trained_frames, n)

    # Add label banners
    r_labeled = [_add_label(f, "RANDOM POLICY",  (180, 60, 60))  for f in r_frames]
    t_labeled = [_add_label(f, "TRAINED POLICY", (60, 150, 100)) for f in t_frames]

    sep = np.zeros((r_labeled[0].shape[0], separator_px, 3), dtype=np.uint8)
    combined = [np.hstack([r, sep, t]) for r, t in zip(r_labeled, t_labeled)]

    imageio.mimsave(output_path, combined, fps=fps, loop=0)
    print(f"Saved comparison GIF: {output_path}")


def _add_label(frame: np.ndarray, text: str, color: tuple) -> np.ndarray:
    """Add colored text banner at top of frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        banner_h = 22
        draw.rectangle([0, 0, frame.shape[1], banner_h], fill=(20, 20, 30))
        draw.text((6, 4), text, fill=color)
        return np.array(img)
    except Exception:
        return frame  # Silently skip label if PIL fails


# ── Eval loop ─────────────────────────────────────────────────────────────────

def evaluate(
    model: PPO,
    env: NavEnv,
    renderer: NavEnvRenderer,
    n_episodes: int = 20,
    command: str = "navigate to the target",
) -> dict:
    """
    Run n_episodes, collect metrics, return best episode frames.
    """
    results = []
    best_episode = None
    best_reward = -float("inf")

    for ep in range(n_episodes):
        ep_data = run_episode(env, model, renderer, command=command)
        results.append(ep_data)

        tag = "SUCCESS" if ep_data["success"] else "fail"
        print(
            f"  ep {ep+1:3d}/{n_episodes} | {tag:7s} | "
            f"reward={ep_data['reward']:+7.1f} | "
            f"dist={ep_data['distance']:.2f}m | "
            f"steps={ep_data['steps']:4d}"
        )

        if ep_data["success"] and ep_data["reward"] > best_reward:
            best_reward = ep_data["reward"]
            best_episode = ep_data

    # Fall back to highest-reward episode if none succeeded
    if best_episode is None:
        best_episode = max(results, key=lambda x: x["reward"])

    metrics = {
        "n_episodes":    n_episodes,
        "success_rate":  float(np.mean([r["success"] for r in results])),
        "mean_reward":   float(np.mean([r["reward"] for r in results])),
        "std_reward":    float(np.std([r["reward"] for r in results])),
        "mean_distance": float(np.mean([r["distance"] for r in results])),
        "mean_steps":    float(np.mean([r["steps"] for r in results])),
        "best_reward":   best_reward,
    }

    return metrics, best_episode["frames"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Record LangNav demo GIFs")
    parser.add_argument("--model",       type=str, required=True,
                        help="Path to PPO checkpoint (.zip)")
    parser.add_argument("--output-dir",  type=str, default="demo")
    parser.add_argument("--n-episodes",  type=int, default=20)
    parser.add_argument("--fps",         type=int, default=15)
    parser.add_argument("--command",     type=str,
                        default="navigate to the target object")
    parser.add_argument("--no-compare",  action="store_true",
                        help="Skip random vs trained comparison GIF")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    env      = NavEnv(max_episode_steps=500, n_obstacles=4)
    renderer = NavEnvRenderer(canvas_px=512, trail_len=80)

    # ── Evaluate ──────────────────────────────────────────────────────────
    print(f"\nEvaluating {args.n_episodes} episodes...")
    metrics, best_frames = evaluate(
        model, env, renderer,
        n_episodes=args.n_episodes,
        command=args.command,
    )

    # ── Save GIF ──────────────────────────────────────────────────────────
    gif_path = os.path.join(args.output_dir, "demo_best.gif")
    imageio.mimsave(gif_path, best_frames, fps=args.fps, loop=0)
    print(f"\nSaved best episode GIF: {gif_path}  ({len(best_frames)} frames)")

    # ── Save MP4 ──────────────────────────────────────────────────────────
    mp4_path = os.path.join(args.output_dir, "demo_best.mp4")
    try:
        import imageio.v3 as iio
        iio.imwrite(mp4_path, best_frames, fps=args.fps, codec="libx264")
        print(f"Saved best episode MP4: {mp4_path}")
    except Exception as e:
        print(f"MP4 skipped (install imageio[ffmpeg] or imageio[pyav]): {e}")

    # ── Random vs trained comparison ──────────────────────────────────────
    if not args.no_compare:
        print("\nRecording random policy episode for comparison...")
        renderer.reset()
        random_data = run_episode(env, model=None, renderer=renderer, command=args.command)

        compare_path = os.path.join(args.output_dir, "demo_compare.gif")
        make_comparison_gif(
            random_frames  = random_data["frames"],
            trained_frames = best_frames,
            output_path    = compare_path,
            fps            = args.fps,
        )

    # ── Save metrics JSON ─────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x))

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Episodes:      {metrics['n_episodes']}")
    print(f"  Success Rate:  {metrics['success_rate']:.1%}")
    print(f"  Mean Reward:   {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Mean Distance: {metrics['mean_distance']:.2f} m")
    print(f"  Mean Steps:    {metrics['mean_steps']:.0f}")
    print("=" * 50)
    print(f"\nOutputs written to: {args.output_dir}/")

    renderer.close()


if __name__ == "__main__":
    main()
