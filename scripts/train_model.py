"""Train PPO navigation agent."""

import argparse
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "langnav/ros2_ws/src/langnav_robot"))

from langnav_rl import PPOTrainer

ENV_BACKENDS = {
    "simple": "langnav_rl.NavEnv",
    "gazebo": "langnav_sim.GazeboEnv",
}


def _load_env_class(backend: str):
    module_path, class_name = ENV_BACKENDS[backend].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Train LangNav PPO agent")
    parser.add_argument("--config",          type=str,  default="configs/ppo_nav.yaml")
    parser.add_argument("--total-timesteps", type=int,  default=None)
    parser.add_argument("--checkpoint-dir",  type=str,  default="checkpoints")
    parser.add_argument("--run-name",        type=str,  default=None)
    parser.add_argument("--wandb",           action="store_true")
    parser.add_argument("--eval-only",       type=str,  default=None,
                        help="Skip training, eval checkpoint at this path")
    parser.add_argument(
        "--backend",
        choices=["simple", "gazebo"],
        default="simple",
        help="'simple' = pure Python sim; 'gazebo' = live Gazebo (needs ROS2)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    total_timesteps = args.total_timesteps or cfg.get("total_timesteps", 1_000_000)
    run_name        = args.run_name        or cfg.get("run_name", "langnav_ppo")
    env_class       = _load_env_class(args.backend)

    print(f"Backend: {args.backend}  |  Run: {run_name}  |  Steps: {total_timesteps:,}")

    trainer = PPOTrainer(
        env_class    = env_class,
        env_kwargs   = cfg.get("env", {}),
        policy       = cfg.get("policy", "MlpPolicy"),
        learning_rate= cfg.get("learning_rate", 3e-4),
        n_steps      = cfg.get("n_steps", 2048),
        batch_size   = cfg.get("batch_size", 64),
        n_epochs     = cfg.get("n_epochs", 10),
        gamma        = cfg.get("gamma", 0.99),
        gae_lambda   = cfg.get("gae_lambda", 0.95),
        clip_range   = cfg.get("clip_range", 0.2),
        ent_coef     = cfg.get("entropy_coef", 0.01),
        n_envs       = cfg.get("n_envs", 4),
        use_wandb    = args.wandb,
    )

    if args.eval_only:
        trainer.load_model(args.eval_only)
        trainer.evaluate(n_episodes=50)
        return

    trainer.train(
        total_timesteps = total_timesteps,
        checkpoint_dir  = args.checkpoint_dir,
        run_name        = run_name,
    )

    print("\nFinal evaluation:")
    trainer.evaluate(n_episodes=50)


if __name__ == "__main__":
    main()
