"""Baseline del planner en un layout (nuevo o conocido) vs los 3 companeros de G8.

Primer paso del playbook de dia-de-competencia (scripts/prepare_new_layout.sh):
valida que el layout carga y que el planner produce sopas ANTES de gastar horas
de entrenamiento. Imprime una tabla y guarda un JSON.

Uso:
  python -m scripts.planner_baseline --layout-file configs/layouts/nuevo.layout
  python -m scripts.planner_baseline --layout cramped_room
Salida JSON (--out, default outputs/dayof/<key>/baseline.json):
  {"layout", "por_companero": {pk: {"score", "soups", "timeouts"}}, "ok": bool}
`ok` = el planner hace >=1 sopa promedio vs greedy Y vs random (si no, arreglar el
planner antes de entrenar). Exit code 0 si ok, 1 si no.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluation import gate_configs as gc
from evaluation.rollout import run_rollouts

REPO = Path(__file__).resolve().parent.parent
SEEDS = json.loads((REPO / "evaluation" / "gate_seeds.json").read_text())["gate_seeds"]
PARTNERS = ["greedy", "greedy_eps", "random_motion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default=None)
    ap.add_argument("--layout-file", default=None)
    ap.add_argument("--out", default=None, help="ruta del baseline.json")
    args = ap.parse_args()
    if not (args.layout or args.layout_file):
        ap.error("se requiere --layout o --layout-file")

    key = Path(args.layout_file).stem if args.layout_file else args.layout
    layout = args.layout or key
    out = Path(args.out) if args.out else REPO / "outputs" / "dayof" / key / "baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    detail = {}
    print(f"[baseline] planner en {key} (gate_seeds x swap):")
    for pk in PARTNERS:
        cfg = gc.make_config(layout, args.layout_file, gc.planner_agent(), gc.partner(pk))
        res = run_rollouts(cfg, seeds=[int(s) for s in SEEDS], swaps=[False, True],
                           test_agent_key="agent_0")
        eps = res["episodes"]
        detail[pk] = {
            "score": round(float(np.mean([e["score"] for e in eps])), 1),
            "soups": round(float(np.mean([e["soups"] for e in eps])), 2),
            "timeouts": int(sum(e["timeouts"] for e in eps)),
        }
        d = detail[pk]
        print(f"  vs {pk:14s}: score={d['score']:>9.1f}  sopas={d['soups']:.2f}  timeouts={d['timeouts']}")

    ok = detail["greedy"]["soups"] >= 1.0 and detail["random_motion"]["soups"] >= 1.0
    result = {"layout": key, "layout_file": args.layout_file, "por_companero": detail, "ok": ok}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not ok:
        print("[baseline] ALERTA: el planner NO llega a 1 sopa promedio -> revisar el "
              "planner/layout antes de entrenar.", file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
