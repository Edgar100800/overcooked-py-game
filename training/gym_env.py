"""Entorno Gymnasium single-agent para PPO (PLAN.md §5).

El OvercookedEnv oficial se envuelve como entorno de UN agente: el companero
(muestreado de la poblacion) actua DENTRO de `step()`. La observacion es el encoding
`lossless_grid` (C,H,W) float32 -> computable desde mdp+state, igual que hara el
StudentAgent desplegado (sin necesidad de mlam ni del env). El indice del agente
(0/1) se randomiza en cada `reset()` para aprender ambos roles.

Fin de episodio = horizon -> `truncated=True` (limite de tiempo), `terminated=False`.
El coeficiente de shaping (`self.coef`) lo setea un callback segun el progreso global.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from src.constants import action_index_to_overcooked_action
from src.environment import build_mdp
from training.partner_population import PartnerPopulation


class SingleAgentOvercooked(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict, partner_weights: dict | None = None,
                 selfplay_models: list | None = None, seed: int | None = None,
                 initial_coef: float = 1.0):
        super().__init__()
        self.env_config = dict(env_config)
        self.mdp = build_mdp(self.env_config)
        self.horizon = int(self.env_config.get("horizon", 400))
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=self.horizon, info_level=0)
        self.population = PartnerPopulation(partner_weights, selfplay_models, seed=seed)
        self.coef = float(initial_coef)

        self.base_env.reset()
        enc = self.mdp.lossless_state_encoding(self.base_env.state)[0]
        h, w, c = np.asarray(enc).shape
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(c, h, w), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        self.rng = np.random.default_rng(seed)
        self.agent_index = 0
        self.partner_index = 1
        self.partner = None
        self.partner_kind = None
        self._ep_sparse = 0.0
        self._ep_shaped = 0.0

    # --- coef seteado por callback (progreso global) ---
    def set_coef(self, coef: float):
        self.coef = float(coef)

    def _encode(self, state, idx) -> np.ndarray:
        enc = self.mdp.lossless_state_encoding(state)[idx]
        return np.ascontiguousarray(np.asarray(enc, dtype=np.float32).transpose(2, 0, 1))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.base_env.reset()
        self.agent_index = int(self.rng.integers(0, 2))
        self.partner_index = 1 - self.agent_index
        self.partner, self.partner_kind = self.population.sample()
        # OJO: reset() ANTES de set_agent_index — Agent.reset() de overcooked borra
        # agent_index, asi que setearlo despues (patron de runner.py).
        if hasattr(self.partner, "reset"):
            self.partner.reset()
        if hasattr(self.partner, "set_mdp"):
            self.partner.set_mdp(self.mdp)
        if hasattr(self.partner, "set_agent_index"):
            self.partner.set_agent_index(self.partner_index)
        self._ep_sparse = 0.0
        self._ep_shaped = 0.0
        obs = self._encode(self.base_env.state, self.agent_index)
        return obs, {"agent_index": self.agent_index, "partner_kind": self.partner_kind}

    def step(self, action):
        our_action = action_index_to_overcooked_action(int(action))
        state = self.base_env.state
        p_out = self.partner.action(state)
        p_action = p_out[0] if isinstance(p_out, tuple) else p_out

        joint = [None, None]
        joint[self.agent_index] = our_action
        joint[self.partner_index] = p_action
        next_state, reward, done, info = self.base_env.step(tuple(joint))

        shaped_by_agent = info.get("shaped_r_by_agent", [0.0, 0.0])
        shaped = float(shaped_by_agent[self.agent_index]) if shaped_by_agent is not None else 0.0
        sparse = float(reward)
        r = sparse + self.coef * shaped
        self._ep_sparse += sparse
        self._ep_shaped += shaped

        obs = self._encode(next_state, self.agent_index)
        terminated = False
        truncated = bool(done)
        info_out = {"sparse": sparse, "shaped": shaped, "coef": self.coef,
                    "agent_index": self.agent_index, "partner_kind": self.partner_kind}
        if done:
            info_out["episode_sparse"] = self._ep_sparse
            info_out["episode_shaped"] = self._ep_shaped
            info_out["episode_soups"] = int(round(self._ep_sparse / 20.0))
        return obs, r, terminated, truncated, info_out


# --------------------------------------------------------------- factory
def make_env(env_config, partner_weights=None, selfplay_models=None, seed=None):
    def _thunk():
        return SingleAgentOvercooked(env_config, partner_weights, selfplay_models, seed=seed)
    return _thunk


# ------------------------------------------------------- G5: check_env report
def _mean_shaped(policy_fn, env_config, partner_weights, n_eps, seed) -> float:
    e = SingleAgentOvercooked(env_config, partner_weights=partner_weights, seed=seed)
    tot = 0.0
    for _ in range(n_eps):
        obs, _ = e.reset()
        done = False
        while not done:
            obs, r, term, trunc, info = e.step(policy_fn(obs, e))
            tot += info["shaped"]
            done = term or trunc
    return tot / n_eps


def make_check_env_report(layout_name: str = "cramped_room", timesteps: int = 150000) -> dict:
    """G5 (PLAN.md §11): check_env pasa; 3 episodios random sin excepciones; PPO
    corre sin NaN y el shaped promedio SUBE vs politica aleatoria.

    Es un SMOKE de que el pipeline aprende, no un benchmark. Por eso se usan ajustes
    favorables (§12-C): companero greedy FIJO, shaping fijo 1.0 (sin annealing),
    mas exploracion (ent 0.05). La robustez con poblacion se valida en G6/G7 (A100).
    Comparacion justa: trained (estocastico) vs random, mismo companero greedy.
    """
    import numpy as np
    from stable_baselines3.common.env_checker import check_env

    env_config = {"layout_name": layout_name, "layout_file": None,
                  "horizon": 200, "old_dynamics": True}
    greedy_only = {"greedy": 1.0}
    report = {"layout": layout_name, "timesteps": timesteps}

    # 1) check_env de SB3
    try:
        check_env(SingleAgentOvercooked(env_config, seed=0), warn=True)
        report["check_env_ok"] = True
    except Exception as exc:
        report["check_env_ok"] = False
        report["check_env_error"] = repr(exc)
        return report

    # 2) 3 episodios aleatorios sin excepciones + shaped medio (companero greedy)
    try:
        random_shaped = _mean_shaped(lambda obs, e: e.action_space.sample(),
                                    env_config, greedy_only, n_eps=6, seed=123)
        report["no_nan"] = True
        report["random_shaped_mean"] = random_shaped
    except Exception as exc:
        report["no_nan"] = False
        report["random_error"] = repr(exc)
        return report

    # 3) PPO (shaping fijo 1.0, ent 0.05, companero greedy) y comparar shaped
    try:
        from training.train_ppo import build_ppo
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        venv = VecMonitor(DummyVecEnv(
            [make_env(env_config, partner_weights=greedy_only, seed=1 + i) for i in range(4)]))
        # coef fijo 1.0 -> sin ShapingAnnealCallback; el env arranca en initial_coef=1.0
        model = build_ppo(venv, obs_kind="lossless_grid", device="cpu", seed=1,
                          n_steps=200, ent_coef=0.05)
        model.learn(total_timesteps=timesteps)
        trained_shaped = _mean_shaped(
            lambda obs, e: int(model.predict(obs, deterministic=False)[0]),
            env_config, greedy_only, n_eps=8, seed=7)
        report["trained_shaped_mean"] = trained_shaped
        report["shaped_improves"] = bool(trained_shaped > random_shaped)
    except Exception as exc:
        report["shaped_improves"] = False
        report["ppo_error"] = repr(exc)
    return report
