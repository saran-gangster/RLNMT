# wandb_reward_callback.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

from transformers import TrainerCallback


@dataclass
class WandbRewardCallback(TrainerCallback):
    reward_fn: object
    log_every: int = 1         # log every optimizer step
    ema_alpha: float = 0.05    # for a smooth "reward increasing" curve

    _ema: Optional[float] = None

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step == 0:
            return
        if self.log_every > 1 and (state.global_step % self.log_every != 0):
            return

        last_log: Optional[Dict[str, float]] = getattr(self.reward_fn, "last_log", None)
        if not last_log:
            return

        import wandb

        # EMA smoothing (this is the one you’ll love watching)
        cur = float(last_log.get("reward/total_mean", 0.0))
        if self._ema is None:
            self._ema = cur
        else:
            self._ema = (1.0 - self.ema_alpha) * self._ema + self.ema_alpha * cur

        payload = dict(last_log)
        payload["reward/total_ema"] = float(self._ema)

        wandb.log(payload, step=state.global_step)