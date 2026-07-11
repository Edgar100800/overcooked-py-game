"""Genera el dataset de behavior cloning desde el PLANNER (STEP A-3, PLAN §7).

El planner SI sabe completar el ciclo en solitario (5 sopas vs random). Este script lo
corre en el layout objetivo contra companeros mixtos (mayoria NO cooperativos) y guarda
pares (obs lossless (C,H,W), accion 0..5) del planner. Ese dataset se usa para pre-entrenar
la politica PPO por imitacion (--bc-data en train_ppo.py), implantando la habilidad solo.

Uso:
  python -m training.collect_bc_data --layout-file configs/layouts/custom_zigzag_kitchen.layout \
      --episodes 400 --out data/bc/custom_zigzag_kitchen.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from policies.basic_policies import StayPolicy, RandomMotionPolicy, GreedyFullTaskPolicy
from policies.planner_agent import PlannerAgent
from src.constants import action_index_to_overcooked_action
from src.environment import build_mdp

# Mayoria no-cooperativa: el dataset debe estar lleno de "planner soloneando".
PARTNER_MIX = [("stay", 0.4), ("random_motion", 0.4), ("greedy", 0.2)]


def _sample_partner(rng):
    r = rng.random()
    acc = 0.0
    for kind, w in PARTNER_MIX:
        acc += w
        if r < acc:
            break
    seed = int(rng.integers(0, 2**31 - 1))
    if kind == "stay":
        return StayPolicy()
    if kind == "random_motion":
        return RandomMotionPolicy(seed=seed)
    return GreedyFullTaskPolicy(seed=seed)


def collect(layout: str | None, layout_file: str | None, episodes: int,
            horizon: int, seed: int, out: Path) -> dict:
    env_config = {"layout_name": None if layout_file else layout,
                  "layout_file": layout_file, "horizon": horizon, "old_dynamics": True}
    mdp = build_mdp(env_config)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    rng = np.random.default_rng(seed)

    all_obs: list[np.ndarray] = []
    all_act: list[int] = []
    soups_total = 0

    for ep in range(episodes):
        env.reset()
        planner = PlannerAgent({"seed": int(rng.integers(0, 2**31 - 1))})
        planner.reset()
        our_idx = int(rng.integers(0, 2))          # ambos roles, como en el gym env
        partner = _sample_partner(rng)
        # patron runner.py: reset -> set_mdp -> set_agent_index (reset borra agent_index)
        if hasattr(partner, "reset"):
            partner.reset()
        partner.set_mdp(mdp)
        partner.set_agent_index(1 - our_idx)

        done = False
        while not done:
            state = env.state
            obs = {"state": state, "mdp": mdp, "agent_index": our_idx}
            a_idx = planner.act(obs)
            enc = np.asarray(mdp.lossless_state_encoding(state)[our_idx], dtype=np.uint8)
            all_obs.append(enc.transpose(2, 0, 1))   # (C,H,W), mismos ejes que el gym env
            all_act.append(int(a_idx))

            p_out = partner.action(state)
            p_action = p_out[0] if isinstance(p_out, tuple) else p_out
            joint = [None, None]
            joint[our_idx] = action_index_to_overcooked_action(a_idx)
            joint[1 - our_idx] = p_action
            _, reward, done, _ = env.step(tuple(joint))
            soups_total += int(round(float(reward) / 20.0))

    obs_arr = np.stack(all_obs).astype(np.uint8)     # uint8: valores del encoding son enteros chicos
    act_arr = np.asarray(all_act, dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=obs_arr, actions=act_arr)
    stats = {
        "samples": int(len(act_arr)),
        "episodes": episodes,
        "soups_total": soups_total,
        "soups_per_ep": soups_total / max(1, episodes),
        "obs_shape": list(obs_arr.shape[1:]),
        "action_hist": {int(a): int((act_arr == a).sum()) for a in range(6)},
        "out": str(out),
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default=None)
    ap.add_argument("--layout-file", default=None)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.layout and not args.layout_file:
        raise SystemExit("--layout o --layout-file requerido")

    import json
    stats = collect(args.layout, args.layout_file, args.episodes, args.horizon,
                    args.seed, Path(args.out))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
