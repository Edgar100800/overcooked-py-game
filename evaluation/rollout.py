"""Driver de rollout para gates (PLAN.md §10).

Corre episodios REALES en el OvercookedEnv oficial (nada de mocks) y captura un
step log crudo con la VERDAD del entorno: sparse reward por step (== entregas de
sopa) y, para el agente bajo prueba, la latencia (elapsed_ms) y los timeouts del
SafeActionWrapper. `run_gate.py` y `verify.py` consumen este log.

Replica el loop de `src/runner.py` (build_env, ObservationBuilder, build_two_policies,
AgentPair) pero registra mas informacion por step de la que guarda el logger normal.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from overcooked_ai_py.agents.agent import AgentPair

from src.constants import overcooked_action_to_index
from src.environment import build_env
from src.observations import ObservationBuilder
from src.policy_loader import build_two_policies
from src.runner import set_global_seed
from evaluation.official_score import official_score, DEFAULT_DELIVERY_REWARD


def _find_attr_in_chain(agent, attr: str):
    """Busca un atributo (p.ej. timeout_count) recorriendo la cadena de wrappers."""
    seen = set()
    cur = agent
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, attr):
            return getattr(cur, attr)
        cur = getattr(cur, "base_agent", None)
    return None


def _detect_test_key(config: dict[str, Any]) -> str:
    """El agente bajo prueba es la politica python_class (planner o student)."""
    policies = config.get("policies", {}) or {}
    for key in ("agent_0", "agent_1"):
        if str(policies.get(key, {}).get("type", "builtin")).lower() == "python_class":
            return key
    # Si ambos son builtin (p.ej. G0 smoke), evaluamos agent_0 por convencion.
    return "agent_0"


def get_delivery_reward(env) -> float:
    """Recompensa por entrega del mdp (para convertir sparse->sopas). Fallback 20."""
    for attr in ("delivery_reward",):
        val = getattr(env.mdp, attr, None)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return DEFAULT_DELIVERY_REWARD


def run_rollouts(
    config: dict[str, Any],
    seeds: list[int],
    swaps: list[bool],
    test_agent_key: str | None = None,
) -> dict[str, Any]:
    """Corre len(seeds) x len(swaps) episodios y devuelve resultados por episodio.

    Returns dict con:
      - delivery_reward
      - horizon
      - episodes: lista de dicts con {seed, role_swap, score, soups, t_first, t_last,
        timeouts, latency_ms (lista), steps (step log crudo)}
    """
    env = build_env(config["environment"])
    horizon = int(config["environment"].get("horizon", env.horizon))
    obs_builder = ObservationBuilder(env, config.get("observation", {}))
    delivery_reward = get_delivery_reward(env)
    if test_agent_key is None:
        test_agent_key = _detect_test_key(config)

    episodes = []
    for seed in seeds:
        for role_swap in swaps:
            ep = _run_one(config, env, obs_builder, int(seed), bool(role_swap),
                          test_agent_key, horizon, delivery_reward)
            episodes.append(ep)

    return {
        "delivery_reward": delivery_reward,
        "horizon": horizon,
        "test_agent_key": test_agent_key,
        "layout_name": env.mdp.layout_name,
        "episodes": episodes,
    }


def _seed_builtin_partners(config: dict, seed: int) -> dict:
    """Inyecta una semilla DETERMINISTA en los partners builtin sin seed propio.

    build_builtin_agent usa policy_config['seed'] (no el seed derivado del loader),
    asi que sin esto los partners random/greedy quedan sin semilla (OS entropy) y no
    son reproducibles -> rompe la comparacion student-vs-planner de G8 y el verify.
    La semilla se deriva del episodio: diversa por gate_seed, identica entre corridas.
    """
    cfg = copy.deepcopy(config)
    for key, offset in (("agent_0", 1000), ("agent_1", 2000)):
        pc = cfg.get("policies", {}).get(key, {})
        if str(pc.get("type", "builtin")).lower() == "builtin" and "seed" not in pc:
            pc["seed"] = int(seed) + offset
    return cfg


def _run_one(config, env, obs_builder, seed, role_swap, test_agent_key,
             horizon, delivery_reward) -> dict[str, Any]:
    set_global_seed(seed)
    config = _seed_builtin_partners(config, seed)
    agent0, agent1 = build_two_policies(config, env, obs_builder, seed=seed)

    # test_agent es el objeto de nuestra politica; su posicion depende del swap.
    if test_agent_key == "agent_0":
        test_agent = agent0
        test_pos = 1 if role_swap else 0
    else:
        test_agent = agent1
        test_pos = 0 if role_swap else 1

    if role_swap:
        agent0, agent1 = agent1, agent0

    agent_pair = AgentPair(agent0, agent1)
    env.reset(regen_mdp=False)
    agent_pair.reset()
    agent_pair.set_mdp(env.mdp)

    timeouts_before = _find_attr_in_chain(test_agent, "timeout_count") or 0
    invalids_before = _find_attr_in_chain(test_agent, "invalid_count") or 0

    steps: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    done = False
    info: dict[str, Any] = {}

    while not done:
        state = env.state
        joint_action_and_infos = agent_pair.joint_action(state)
        joint_action, joint_infos = zip(*joint_action_and_infos)
        next_state, reward, done, info = env.step(joint_action, joint_infos)

        our_info = joint_infos[test_pos] if test_pos < len(joint_infos) else {}
        elapsed = our_info.get("elapsed_ms")
        if elapsed is not None:
            latency_ms.append(float(elapsed))
        sparse_by_agent = info.get("sparse_r_by_agent") if isinstance(info, dict) else None

        steps.append({
            "timestep": int(state.timestep),
            "sparse": float(reward),
            "sparse_r_by_agent": list(sparse_by_agent) if sparse_by_agent is not None else None,
            "our_elapsed_ms": None if elapsed is None else float(elapsed),
            "our_timeout": bool(our_info.get("timeout_action_replaced", False)),
            "our_invalid": bool(our_info.get("invalid_action_replaced", False)),
        })

    timeouts_after = _find_attr_in_chain(test_agent, "timeout_count") or 0
    invalids_after = _find_attr_in_chain(test_agent, "invalid_count") or 0
    timeouts = int(timeouts_after - timeouts_before)
    invalids = int(invalids_after - invalids_before)

    sb = official_score(
        [s["sparse"] for s in steps], timeouts, horizon, delivery_reward,
        timesteps=[s["timestep"] for s in steps],
    )

    lat = np.array(latency_ms) if latency_ms else np.array([0.0])
    return {
        "seed": seed,
        "role_swap": role_swap,
        **sb.to_dict(),
        "invalids": invalids,
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p99": float(np.percentile(lat, 99)),
        "latency_max": float(lat.max()),
        "n_steps": len(steps),
        "steps": steps,
    }
