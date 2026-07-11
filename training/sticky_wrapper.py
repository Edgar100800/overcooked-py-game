"""StickyActionWrapper: un partner que repite su accion previa con prob p.

Modela el "sticky actions" del escenario 2 (PLAN.md §1). Envuelve a cualquier
partner con interfaz `action(state) -> overcooked_action`.
"""

from __future__ import annotations

import numpy as np


class StickyPartner:
    def __init__(self, base_partner, stick_prob: float = 0.25, seed=None):
        self.base = base_partner
        self.stick_prob = float(stick_prob)
        self.rng = np.random.default_rng(seed)
        self._last_action = None

    def set_mdp(self, mdp):
        if hasattr(self.base, "set_mdp"):
            self.base.set_mdp(mdp)

    def set_agent_index(self, idx):
        if hasattr(self.base, "set_agent_index"):
            self.base.set_agent_index(idx)

    def reset(self):
        self._last_action = None
        if hasattr(self.base, "reset"):
            self.base.reset()

    def action(self, state):
        base_out = self.base.action(state)
        action = base_out[0] if isinstance(base_out, tuple) else base_out
        if self._last_action is not None and self.rng.random() < self.stick_prob:
            action = self._last_action
        self._last_action = action
        return action, {"policy_name": "sticky"}
