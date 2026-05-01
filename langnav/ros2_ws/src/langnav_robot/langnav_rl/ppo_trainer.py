"""PPO training loop for navigation policy."""

import os
from typing import Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from .nav_env import NavEnv


class PPOTrainer:
    """Train PPO agent for robot navigation."""

    def __init__(
        self,
        env_kwargs: Dict[str, Any] = None,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        use_wandb: bool = False,
    ):
        """
        Initialize PPO trainer.

        Args:
            env_kwargs: Environment parameters
            policy: Policy network architecture
            learning_rate: PPO learning rate
            n_steps: Steps per rollout
            batch_size: Training batch size
            use_wandb: Enable W&B logging
        """
        self.env_kwargs = env_kwargs or {}
        self.policy = policy
        self.lr = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.use_wandb = use_wandb

        self.env = NavEnv(**self.env_kwargs)
        self.model = None

    def train(
        self,
        total_timesteps: int = 1_000_000,
        checkpoint_dir: str = "checkpoints",
        run_name: str = "langnav_ppo",
    ):
        """
        Train PPO agent.

        Args:
            total_timesteps: Total training timesteps
            checkpoint_dir: Where to save checkpoints
            run_name: W&B run name (if enabled)
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        callbacks = []

        # Checkpointing callback
        checkpoint_cb = CheckpointCallback(
            save_freq=10000,
            save_path=checkpoint_dir,
            name_prefix="ppo_nav",
        )
        callbacks.append(checkpoint_cb)

        # W&B callback
        if self.use_wandb:
            wandb.init(project="langnav", name=run_name)
            wandb_cb = WandbCallback(
                model_save_path=checkpoint_dir,
                verbose=0,
            )
            callbacks.append(wandb_cb)

        # Create and train model
        self.model = PPO(
            self.policy,
            self.env,
            learning_rate=self.lr,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            verbose=1,
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )

        # Save final model
        final_path = os.path.join(checkpoint_dir, "ppo_nav_final")
        self.model.save(final_path)
        print(f"Model saved to {final_path}")

        if self.use_wandb:
            wandb.finish()

    def evaluate(self, n_episodes: int = 10):
        """
        Evaluate trained agent.

        Args:
            n_episodes: Number of episodes to run

        Returns:
            Mean episode reward
        """
        if self.model is None:
            raise ValueError("No trained model. Call train() first.")

        total_reward = 0
        success_count = 0

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

                if info.get("success", False):
                    success_count += 1

            total_reward += episode_reward

        mean_reward = total_reward / n_episodes
        success_rate = success_count / n_episodes

        print(f"Mean Reward: {mean_reward:.2f}, Success Rate: {success_rate:.2%}")
        return mean_reward

    def load_model(self, path: str):
        """Load pre-trained model."""
        self.model = PPO.load(path, env=self.env)
