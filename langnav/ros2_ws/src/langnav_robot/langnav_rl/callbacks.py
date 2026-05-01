"""
Custom SB3 callbacks for LangNav training.

EpisodeMetricsCallback  — logs per-episode stats (success_rate, mean_dist, etc.)
WandbSummaryCallback    — pushes live metrics to W&B every N rollouts
"""

import numpy as np
from collections import deque
from typing import Deque, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """
    Track per-episode success rate, distance, and episode length
    over a sliding window. Injects metrics into logger for SB3 + W&B.
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        """
        Args:
            window: Rolling window size for averaging metrics
            verbose: SB3 verbosity level
        """
        super().__init__(verbose)
        self._window = window
        self._successes:      Deque[float] = deque(maxlen=window)
        self._distances:      Deque[float] = deque(maxlen=window)
        self._episode_lengths: Deque[int]  = deque(maxlen=window)

    def _on_step(self) -> bool:
        # SB3 stores infos in self.locals["infos"] after each env step
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_lengths.append(info["episode"]["l"])
                self._successes.append(float(info.get("success", False)))
                self._distances.append(float(info.get("distance", 0.0)))

        return True

    def _on_rollout_end(self) -> None:
        """Log rolling averages at end of every rollout."""
        if not self._successes:
            return

        self.logger.record("metrics/success_rate",   np.mean(self._successes))
        self.logger.record("metrics/mean_dist",      np.mean(self._distances))
        self.logger.record("metrics/mean_ep_length", np.mean(self._episode_lengths) if self._episode_lengths else 0)
        self.logger.record("metrics/episodes_seen",  len(self._successes))

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._successes)) if self._successes else 0.0


class WandbSummaryCallback(BaseCallback):
    """
    Push training summary to W&B: model artifact + final metrics table.
    Triggered once at training end.
    """

    def __init__(self, model_path: str = "checkpoints/ppo_nav_final", verbose: int = 0):
        """
        Args:
            model_path: Where the final model checkpoint is saved
        """
        super().__init__(verbose)
        self.model_path = model_path

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        try:
            import wandb
            if wandb.run is None:
                return

            # Log final model as artifact
            artifact = wandb.Artifact(
                name="ppo_nav_model",
                type="model",
                description="Final PPO navigation policy",
            )
            artifact.add_file(f"{self.model_path}.zip")
            wandb.log_artifact(artifact)

            if self.verbose:
                print(f"[WandbSummaryCallback] Model artifact uploaded: {self.model_path}.zip")
        except Exception as e:
            if self.verbose:
                print(f"[WandbSummaryCallback] W&B upload skipped: {e}")
