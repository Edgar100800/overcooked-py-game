"""Verificacion extra vs companeros STICKY (escenarios 2 y 3 de la competencia).

El enable-check oficial (scripts/enable_model.py) valida vs greedy, greedy_eps y
random_motion, pero NO contra el companero exacto del escenario 2 (greedy_full_task
con sticky actions) ni el del 3 (sticky + random actions). Este script mide eso,
SIN tocar evaluation/ (congelado por freeze.sha256): parchea en runtime
`src.policy_loader.build_builtin_agent` para registrar el builtin "greedy_sticky"
(GreedyFullTaskPolicy envuelto en training/sticky_wrapper.StickyPartner) y corre el
MISMO loop oficial (gate_seeds x swap via evaluation.rollout.run_rollouts).

Uso:
  python -m scripts.check_vs_sticky --layout counter_circuit --agent planner
  python -m scripts.check_vs_sticky --layout coordination_ring --agent student
Salida: outputs/dayof/<key>/sticky_<agent>.json
  {"layout", "agent", "por_companero": {greedy_sticky, greedy_sticky_eps}, ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import src.policy_loader as policy_loader
from evaluation import gate_configs as gc
from evaluation.rollout import run_rollouts
from policies.basic_policies import GreedyFullTaskPolicy
from training.sticky_wrapper import StickyPartner

REPO = Path(__file__).resolve().parent.parent
SEEDS = json.loads((REPO / "evaluation" / "gate_seeds.json").read_text())["gate_seeds"]

_ORIG_BUILD = policy_loader.build_builtin_agent


def _patched_build(name, env, policy_config=None):
    """Como el original, pero con el builtin extra "greedy_sticky"."""
    if str(name).strip().lower() == "greedy_sticky":
        pc = policy_config or {}
        base = GreedyFullTaskPolicy(
            ingredient=pc.get("ingredient", "onion"),
            avoid_teammate=pc.get("avoid_teammate", True),
            seed=pc.get("seed"),
        )
        return StickyPartner(base, stick_prob=pc.get("stick_prob", 0.25), seed=pc.get("seed"))
    return _ORIG_BUILD(name, env, policy_config)


def sticky_partner_cfg(stick_prob: float, eps: float) -> dict:
    c = {"type": "builtin", "name": "greedy_sticky", "ingredient": "onion",
         "avoid_teammate": True, "stick_prob": stick_prob, **gc._WRAP}
    if eps > 0:
        # el wrap oficial (EpsilonActionWrapper) mete las random actions encima
        c["random_action_prob"] = eps
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default=None)
    ap.add_argument("--layout-file", default=None)
    ap.add_argument("--agent", choices=["planner", "student"], default="planner")
    ap.add_argument("--model-path", default=None,
                    help="evalua un candidato PPO explicito (student con require_enabled=False)")
    ap.add_argument("--stick-prob", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=0.15, help="random actions del esc. 3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not (args.layout or args.layout_file):
        ap.error("se requiere --layout o --layout-file")

    key = Path(args.layout_file).stem if args.layout_file else args.layout
    layout = args.layout or key
    out = Path(args.out) if args.out else REPO / "outputs" / "dayof" / key / f"sticky_{args.agent}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    policy_loader.build_builtin_agent = _patched_build

    if args.agent == "planner":
        agent_cfg = gc.planner_agent()
    elif args.model_path:
        agent_cfg = gc.student_agent({"layout": layout, "layout_file": args.layout_file,
                                      "model_path": args.model_path,
                                      "require_enabled": False})
    else:
        agent_cfg = gc.student_agent()
    partners = {
        "greedy_sticky": sticky_partner_cfg(args.stick_prob, 0.0),        # escenario 2
        "greedy_sticky_eps": sticky_partner_cfg(args.stick_prob, args.eps),  # escenario 3
    }

    detail = {}
    print(f"[sticky-check] {args.agent} en {key} (stick={args.stick_prob}, eps={args.eps}):")
    for pk, pcfg in partners.items():
        cfg = gc.make_config(layout, args.layout_file, agent_cfg, pcfg)
        res = run_rollouts(cfg, seeds=[int(s) for s in SEEDS], swaps=[False, True],
                           test_agent_key="agent_0")
        eps_list = res["episodes"]
        detail[pk] = {
            "score": round(float(np.mean([e["score"] for e in eps_list])), 1),
            "soups": round(float(np.mean([e["soups"] for e in eps_list])), 2),
            "timeouts": int(sum(e["timeouts"] for e in eps_list)),
        }
        d = detail[pk]
        print(f"  vs {pk:18s}: score={d['score']:>9.1f}  sopas={d['soups']:.2f}  timeouts={d['timeouts']}")

    result = {"layout": key, "layout_file": args.layout_file, "agent": args.agent,
              "stick_prob": args.stick_prob, "eps": args.eps, "por_companero": detail}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
