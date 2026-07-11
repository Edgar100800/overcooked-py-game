"""enable_model.py — habilita un modelo PPO SOLO si es robusto (no rompe G8).

El gate G7 (congelado) solo compara vs greedy, por eso puede habilitar un modelo que
luego colapsa vs random_motion y rompe G8. Este helper (fuera de evaluation/, no toca el
freeze) hace la decision CORRECTA de despliegue:

  1. Copia el candidato a la ruta canonica models/<key>/best.zip (la que carga el student).
  2. Evalua el student-con-PPO vs el planner puro en los 3 companeros de G8
     (greedy, greedy_eps, random_motion) x swap, con gate_seeds.
  3. Habilita (escribe models/<key>/enabled) SOLO si el student NUNCA empeora al planner
     en ninguna celda (mismo criterio que G8). Si regresa, revierte y reporta.

Uso:
  python -m scripts.enable_model --layout custom_room \
      --layout-file configs/layouts/custom_room.layout \
      --model models/custom_room/seed101/best.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from evaluation import gate_configs as gc
from evaluation.rollout import run_rollouts

REPO = Path(__file__).resolve().parent.parent
SEEDS = json.loads((REPO / "evaluation" / "gate_seeds.json").read_text())["gate_seeds"]
PARTNERS = ["greedy", "greedy_eps", "random_motion"]


def _mean_score(cfg):
    res = run_rollouts(cfg, seeds=[int(s) for s in SEEDS], swaps=[False, True],
                       test_agent_key="agent_0")
    import numpy as np
    return float(np.mean([e["score"] for e in res["episodes"]]))


def evaluate(layout, layout_file, model_path):
    """(robusto, detalle) comparando student-PPO vs planner por companero.

    RACE-SAFE: el student carga el candidato via model_path EXPLICITO (sin tocar la
    ruta canonica models/<key>/best.zip), asi varios enable-checks del mismo layout
    pueden correr en paralelo sin pisarse.
    """
    detail = {}
    robust = True
    scfg = {"layout": layout, "layout_file": layout_file,
            "model_path": str(model_path), "require_enabled": False}
    for pk in PARTNERS:
        student = _mean_score(gc.make_config(layout, layout_file, gc.student_agent(scfg), gc.partner(pk)))
        planner = _mean_score(gc.make_config(layout, layout_file, gc.planner_agent(), gc.partner(pk)))
        ok = student >= planner - 1e-6
        detail[pk] = {"student": round(student, 1), "planner": round(planner, 1), "ok": ok}
        robust = robust and ok
    return robust, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--layout-file", default=None)
    ap.add_argument("--model", required=True, help="ruta al best.zip candidato")
    args = ap.parse_args()

    key = Path(args.layout_file).stem if args.layout_file else args.layout
    robust, detail = evaluate(args.layout, args.layout_file, args.model)
    print(json.dumps({"layout": key, "model": args.model, "robust": robust,
                      "por_companero": detail}, indent=2))

    if robust:
        # Solo al FINAL, y solo si es robusto, se toca la ruta canonica del despliegue.
        model_dir = REPO / "models" / key
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.model, model_dir / "best.zip")
        (model_dir / "enabled").write_text(
            f"HABILITADO (robusto vs {PARTNERS}) modelo={args.model}\n{detail}\n")
        print(f"[enable] {key} HABILITADO: el PPO supera/iguala al planner en TODOS los companeros.")
    else:
        bad = [pk for pk, d in detail.items() if not d["ok"]]
        print(f"[enable] {key} NO habilitado: regresa vs {bad} -> queda en planner (entrega segura).")


if __name__ == "__main__":
    main()
