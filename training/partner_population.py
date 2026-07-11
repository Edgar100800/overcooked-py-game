"""Poblacion de companeros para entrenar PPO robusto (PLAN.md §5).

Sampleo por `reset()`:
  greedy limpio 35% | greedy+sticky+eps 25% | random_motion 20% |
  self-play (checkpoints congelados propios) 15% | stay 5%.

Cada partner expone: set_mdp(mdp), set_agent_index(idx), reset(), action(state).
`action(state)` devuelve una accion de Overcooked (o (accion, info)); el gym env
normaliza. Si no hay checkpoints de self-play, ese peso se reparte a greedy.
"""

from __future__ import annotations

import numpy as np

from overcooked_ai_py.mdp.actions import Action

from policies.basic_policies import StayPolicy, RandomMotionPolicy, GreedyFullTaskPolicy
from training.sticky_wrapper import StickyPartner

DEFAULT_WEIGHTS = {
    "greedy": 0.35,
    "greedy_sticky_eps": 0.25,
    "random_motion": 0.20,
    "self_play": 0.15,
    "stay": 0.05,
}


class _EpsilonPartner:
    """Envuelve un partner y con prob eps toma una accion uniforme (esc. 3)."""
    def __init__(self, base, eps=0.15, seed=None):
        self.base = base
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)
        self._all = list(Action.ALL_ACTIONS)

    def set_mdp(self, mdp):
        if hasattr(self.base, "set_mdp"): self.base.set_mdp(mdp)
    def set_agent_index(self, idx):
        if hasattr(self.base, "set_agent_index"): self.base.set_agent_index(idx)
    def reset(self):
        if hasattr(self.base, "reset"): self.base.reset()

    def action(self, state):
        if self.rng.random() < self.eps:
            return self._all[int(self.rng.integers(0, len(self._all)))], {"eps": True}
        out = self.base.action(state)
        return (out if isinstance(out, tuple) else (out, {}))


class SelfPlayPartner:
    """Companero = un checkpoint PPO propio congelado (self-play)."""
    def __init__(self, model, deterministic=False):
        self.model = model
        self.deterministic = deterministic
        self.mdp = None
        self.agent_index = 0

    def set_mdp(self, mdp): self.mdp = mdp
    def set_agent_index(self, idx): self.agent_index = int(idx)
    def reset(self): pass

    def action(self, state):
        from src.constants import action_index_to_overcooked_action
        enc = self.mdp.lossless_state_encoding(state)[self.agent_index]
        obs = np.asarray(enc, dtype=np.float32).transpose(2, 0, 1)
        act_idx, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action_index_to_overcooked_action(int(act_idx)), {"policy_name": "self_play"}


class PartnerPopulation:
    def __init__(self, weights: dict | None = None, selfplay_models: list | None = None,
                 seed=None):
        # _requested conserva los pesos pedidos: si self_play>0 pero aun no hay
        # checkpoints, el peso va a greedy HASTA que add_selfplay_path() los aporte.
        self._requested = dict(weights or DEFAULT_WEIGHTS)
        self.weights = dict(self._requested)
        self.selfplay_models = selfplay_models or []
        self.rng = np.random.default_rng(seed)
        if not self.selfplay_models and self.weights.get("self_play", 0) > 0:
            self.weights["greedy"] = self.weights.get("greedy", 0) + self.weights["self_play"]
            self.weights["self_play"] = 0.0
        self._recompute()

    def add_selfplay_path(self, path: str, max_pool: int = 6):
        """Carga un checkpoint PPO congelado y lo agrega al pool (FCP-lite).

        Al llegar el primer checkpoint se restauran los pesos pedidos (self_play
        recupera su probabilidad). El pool se capa a max_pool (FIFO) manteniendo
        diversidad de 'edades' del propio agente.
        """
        from stable_baselines3 import PPO
        model = PPO.load(path, device="cpu")
        self.selfplay_models.append(model)
        if len(self.selfplay_models) > max_pool:
            # conservar el mas viejo (companero ~random, fuerza solo) y podar el 2do
            self.selfplay_models.pop(1)
        if self._requested.get("self_play", 0) > 0:
            self.weights = dict(self._requested)
            self._recompute()

    def _recompute(self):
        self._kinds = list(self.weights.keys())
        w = np.array([self.weights[k] for k in self._kinds], dtype=float)
        self._probs = w / w.sum()

    def set_weights(self, weights: dict):
        """Cambia los pesos en caliente (para curriculum). Redistribuye self_play si no hay modelos."""
        self._requested = dict(weights)
        self.weights = dict(weights)
        if not self.selfplay_models and self.weights.get("self_play", 0) > 0:
            self.weights["greedy"] = self.weights.get("greedy", 0) + self.weights["self_play"]
            self.weights["self_play"] = 0.0
        self._recompute()

    def sample(self):
        kind = self._kinds[int(self.rng.choice(len(self._kinds), p=self._probs))]
        seed = int(self.rng.integers(0, 2**31 - 1))
        return self._build(kind, seed), kind

    def _build(self, kind, seed):
        if kind == "stay":
            return StayPolicy()
        if kind == "random_motion":
            return RandomMotionPolicy(seed=seed)
        if kind == "greedy":
            return GreedyFullTaskPolicy(seed=seed)
        if kind == "greedy_sticky_eps":
            base = GreedyFullTaskPolicy(seed=seed)
            sticky = StickyPartner(base, stick_prob=0.25, seed=seed)
            return _EpsilonPartner(sticky, eps=0.15, seed=seed)
        if kind == "self_play":
            model = self.selfplay_models[int(self.rng.integers(0, len(self.selfplay_models)))]
            return SelfPlayPartner(model, deterministic=False)
        return GreedyFullTaskPolicy(seed=seed)
