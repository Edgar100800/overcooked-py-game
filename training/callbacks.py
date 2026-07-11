"""Callbacks de entrenamiento (PLAN.md §5).

- ShapingAnnealCallback: setea el coef de shaping en todos los envs segun el
  progreso GLOBAL (1.0 -> 0.0 en el primer `anneal_frac`).
- OfficialScoreEvalCallback: cada `eval_freq` steps evalua el modelo vs
  greedy_full_task con el SCORE OFICIAL (no el reward de entrenamiento) y guarda
  `best.zip` por score oficial. Es un proxy del harness de gates para seleccionar
  checkpoint; la aprobacion real de G6/G7 la hace evaluation/run_gate.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from training.reward_shaping import shaped_coef
from evaluation.official_score import official_score


class ShapingAnnealCallback(BaseCallback):
    def __init__(self, total_timesteps: int, anneal_frac: float = 0.6, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.anneal_frac = anneal_frac

    def _on_rollout_start(self) -> None:
        coef = shaped_coef(self.num_timesteps, self.total_timesteps, self.anneal_frac)
        try:
            self.training_env.env_method("set_coef", coef)
        except Exception:
            pass

    def _on_step(self) -> bool:
        return True


class OfficialScoreEvalCallback(BaseCallback):
    def __init__(self, eval_env_config: dict, out_dir: str, eval_freq: int = 100000,
                 n_eval_episodes: int = 6, verbose=1):
        super().__init__(verbose)
        self.eval_env_config = eval_env_config
        self.out_dir = Path(out_dir)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_score = -np.inf
        self._eval_env = None

    def _init_callback(self) -> None:
        from training.gym_env import SingleAgentOvercooked
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # partner FIJO greedy para el score oficial (compañero oficial esc. 1)
        self._eval_env = SingleAgentOvercooked(
            self.eval_env_config, partner_weights={"greedy": 1.0}, seed=999)

    def _evaluate(self) -> float:
        scores = []
        env = self._eval_env
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            sparse_per_step, timesteps = [], []
            t = 0
            while not done:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(int(a))
                sparse_per_step.append(info["sparse"])
                timesteps.append(t)
                t += 1
                done = term or trunc
            sb = official_score(sparse_per_step, timeouts=0, horizon=env.horizon,
                                delivery_reward=20.0, timesteps=timesteps)
            scores.append(sb.score)
        return float(np.mean(scores))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            score = self._evaluate()
            self.logger.record("eval/official_score", score)
            if self.verbose:
                print(f"[eval] step={self.num_timesteps} official_score={score:.1f} "
                      f"(best={self.best_score:.1f})")
            if score > self.best_score:
                self.best_score = score
                self.model.save(str(self.out_dir / "best.zip"))
                (self.out_dir / "best_score.txt").write_text(f"{score}\n")
        return True
