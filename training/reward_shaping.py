"""Reward shaping con annealing (PLAN.md §5).

r = sparse + coef * shaped_r_del_agente
Los shaped rewards nativos del MDP (pot=3, dish=3, soup_pickup=5) vienen en
info['shaped_r_by_agent']. `coef` decae linealmente 1.0 -> 0.0 en el primer
`anneal_frac` (default 0.6) del entrenamiento, para que la politica final optimice
el sparse puro (entregas de sopa) y no el shaping.
"""

from __future__ import annotations


def shaped_coef(step: int, total_timesteps: int, anneal_frac: float = 0.6,
                start: float = 1.0, end: float = 0.0) -> float:
    """Coeficiente de shaping en el timestep `step` (por-env, aproximado)."""
    if anneal_frac <= 0:
        return end
    horizon = max(1.0, total_timesteps * anneal_frac)
    frac = min(1.0, step / horizon)
    return float(start + (end - start) * frac)
