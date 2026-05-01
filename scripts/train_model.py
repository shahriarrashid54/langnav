"""Train PPO navigation agent."""

import argparse
import yaml
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "langnav/ros2_ws/src/langnav_robot"))

from langnav_rl import PPOTrainer


ENV_BACKENDS = {
    "simple": "langnav_rl.NavEnv",
    "gazebo": "langnav_sim.GazeboEnv",
}


def main():
    parser = argparse.ArgumentParser(description="Train LangNav PPO agent")
    parser.add_argument("--config",           type=str,  default="configs/ppo_nav.yaml")
    parser.add_argument("--total-timesteps",  type=int,  default=1_000_000)
    parser.add_argument("--checkpoint-dir",   type=str,  default="checkpoints")
    parser.add_argument("--wandb",            action="store_true")
    parser.add_argument(
        "--backend",
        choices=["simple", "gazebo"],
        default="simple",
        help="'simple' = pure Python sim, 'gazebo' = real Gazebo (requires ROS2)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}, Backend: {args.backend}")

    # Resolve env class
    module_path, class_name = ENV_BACKENDS[args.backend].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    env_class = getattr(module, class_name)

    # Create trainer
    trainer = PPOTrainer(
        env_class=env_class,
        env_kwargs=config.get("env", {}),
        policy=config.get("policy", "MlpPolicy"),
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        use_wandb=args.wandb,
    )

    # Train
    print(f"Training for {args.total_timesteps} timesteps...")
    trainer.train(
        total_timesteps=args.total_timesteps,
        checkpoint_dir=args.checkpoint_dir,
        run_name=config.get("run_name", "langnav_ppo"),
    )

    # Evaluate
    print("Evaluating...")
    trainer.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()
