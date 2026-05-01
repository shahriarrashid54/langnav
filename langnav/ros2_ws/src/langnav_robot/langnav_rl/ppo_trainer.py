"""PPO training loop with VecEnv, evaluation, and W&B logging."""

import os
from typing import Dict, Any, Type
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from .nav_env import NavEnv
from .callbacks import EpisodeMetricsCallback, WandbSummaryCallback


class PPOTrainer:
    """Train and evaluate a PPO navigation policy."""

    def __init__(
        self,
        env_class: Type[gym.Env] = NavEnv,
        env_kwargs: Dict[str, Any] = None,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        n_envs: int = 4,
        use_wandb: bool = False,
    ):
        """
        Args:
            env_class: Gymnasium environment class (NavEnv or GazeboEnv)
            env_kwargs: Kwargs forwarded to env constructor
            policy: SB3 policy network ("MlpPolicy" / "CnnPolicy")
            learning_rate: Adam lr
            n_steps: Rollout length per env before update
            batch_size: Minibatch size
            n_epochs: PPO update epochs per rollout
            gamma: Discount factor
            gae_lambda: GAE smoothing
            clip_range: PPO clip epsilon
            ent_coef: Entropy bonus coefficient
            n_envs: Parallel envs for VecEnv
            use_wandb: Enable W&B logging
        """
        self.env_class   = env_class
        self.env_kwargs  = env_kwargs or {}
        self.policy      = policy
        self.n_envs      = n_envs
        self.use_wandb   = use_wandb

        self._ppo_kwargs = dict(
            learning_rate = learning_rate,
            n_steps       = n_steps,
            batch_size    = batch_size,
            n_epochs      = n_epochs,
            gamma         = gamma,
            gae_lambda    = gae_lambda,
            clip_range    = clip_range,
            ent_coef      = ent_coef,
            verbose       = 1,
        )

        self.model: PPO = None
        self._train_env = None
        self._eval_env  = None

    # ── Public API ───────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 1_000_000,
        checkpoint_dir: str = "checkpoints",
        run_name: str = "langnav_ppo",
    ):
        """
        Train PPO agent.

        Args:
            total_timesteps: Training budget in env steps
            checkpoint_dir: Directory for model checkpoints
            run_name: Run identifier (used by W&B and checkpoint prefix)
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._init_wandb(run_name)

        # Vectorized training envs
        self._train_env = make_vec_env(
            self.env_class,
            n_envs=self.n_envs,
            env_kwargs=self.env_kwargs,
            wrapper_class=Monitor,
        )

        # Single eval env
        self._eval_env = Monitor(self.env_class(**self.env_kwargs))

        # Build model
        self.model = PPO(
            self.policy,
            self._train_env,
            tensorboard_log=os.path.join(checkpoint_dir, "tensorboard"),
            **self._ppo_kwargs,
        )

        callbacks = self._build_callbacks(checkpoint_dir, run_name)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        final_path = os.path.join(checkpoint_dir, f"{run_name}_final")
        self.model.save(final_path)
        print(f"Saved: {final_path}.zip")

        self._finish_wandb()

    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """
        Evaluate trained agent, return metrics dict.

        Args:
            n_episodes: Episode count

        Returns:
            {"mean_reward": ..., "success_rate": ..., "mean_distance": ...}
        """
        if self.model is None:
            raise RuntimeError("No trained model. Call train() or load_model() first.")

        env = self.env_class(**self.env_kwargs)
        rewards, successes, distances = [], [], []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated

            rewards.append(ep_reward)
            successes.append(float(info.get("success", False)))
            distances.append(float(info.get("distance", 0.0)))

        metrics = {
            "mean_reward":   float(np.mean(rewards)),
            "std_reward":    float(np.std(rewards)),
            "success_rate":  float(np.mean(successes)),
            "mean_distance": float(np.mean(distances)),
        }

        print(
            f"Eval ({n_episodes} eps) | "
            f"Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f} | "
            f"Success: {metrics['success_rate']:.1%} | "
            f"Dist: {metrics['mean_distance']:.2f}m"
        )

        if self.use_wandb:
            try:
                import wandb
                if wandb.run:
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()})
            except Exception:
                pass

        return metrics

    def load_model(self, path: str):
        """Load pre-trained model from checkpoint path."""
        if self._train_env is None:
            self._train_env = make_vec_env(
                self.env_class, n_envs=1, env_kwargs=self.env_kwargs
            )
        self.model = PPO.load(path, env=self._train_env)
        print(f"Loaded model from {path}")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_callbacks(self, checkpoint_dir: str, run_name: str) -> CallbackList:
        checkpoint_cb = CheckpointCallback(
            save_freq    = max(10_000 // self.n_envs, 1),
            save_path    = checkpoint_dir,
            name_prefix  = run_name,
            verbose      = 0,
        )

        eval_cb = EvalCallback(
            self._eval_env,
            best_model_save_path = os.path.join(checkpoint_dir, "best"),
            log_path             = os.path.join(checkpoint_dir, "eval_logs"),
            eval_freq            = max(20_000 // self.n_envs, 1),
            n_eval_episodes      = 20,
            deterministic        = True,
            verbose              = 1,
        )

        metrics_cb = EpisodeMetricsCallback(window=100, verbose=0)

        cbs = [checkpoint_cb, eval_cb, metrics_cb]

        if self.use_wandb:
            final_path = os.path.join(checkpoint_dir, f"{run_name}_final")
            cbs.append(WandbSummaryCallback(model_path=final_path))

            try:
                from wandb.integration.sb3 import WandbCallback
                cbs.append(WandbCallback(model_save_path=checkpoint_dir, verbose=0))
            except ImportError:
                pass

        return CallbackList(cbs)

    def _init_wandb(self, run_name: str):
        if not self.use_wandb:
            return
        try:
            import wandb
            wandb.init(
                project = "langnav",
                name    = run_name,
                config  = {**self._ppo_kwargs, "n_envs": self.n_envs},
            )
        except Exception as e:
            print(f"W&B init failed ({e}). Continuing without W&B.")
            self.use_wandb = False

    def _finish_wandb(self):
        if not self.use_wandb:
            return
        try:
            import wandb
            if wandb.run:
                wandb.finish()
        except Exception:
            pass
