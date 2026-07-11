"""run_gate.py — ejecuta un gate y emite artefactos (PLAN.md §8-11).

REGLAS ANTI-TRAMPA (§10):
  - Es el UNICO script que escribe goals/progress.json.
  - Corre rollouts REALES en el env oficial con gate_seeds (nunca dev_seeds).
  - verify.py recomputa el score desde el step log crudo (tolerancia 0).
  - Verifica freeze.sha256 al inicio; si evaluation/ cambio tras el freeze, aborta.
  - Emite outputs/gates/GX_<timestamp>/ con results.json + step logs crudos + config.
  Sin artefactos -> gate no aprobado, aunque el numero se haya visto en consola.

Uso:
  python -m evaluation.run_gate --gate G2
  python -m evaluation.run_gate --gate G3 --student planner
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from evaluation import gate_configs as gc
from evaluation.official_score import check_synthetic, DEFAULT_DELIVERY_REWARD
from evaluation.rollout import run_rollouts
from evaluation.verify import verify_rollouts, verify_freeze, write_freeze

REPO = Path(__file__).resolve().parent.parent
GATES_OUT = REPO / "outputs" / "gates"
PROGRESS = REPO / "goals" / "progress.json"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_seeds() -> list[int]:
    data = json.loads((REPO / "evaluation" / "gate_seeds.json").read_text())
    return [int(s) for s in data["gate_seeds"]]


def _write_progress(gate_id: str, passed: bool, metric: dict, artifact_dir: str):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    progress = {}
    if PROGRESS.exists():
        try:
            progress = json.loads(PROGRESS.read_text())
        except Exception:
            progress = {}
    progress[gate_id] = {
        "passed": bool(passed),
        "metric": metric,
        "artifact_dir": artifact_dir,
        "timestamp": _timestamp(),
    }
    PROGRESS.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")


def _run_rollout_gate(gate_id: str, student_kind: str, seeds: list[int], out_dir: Path):
    """Corre las celdas de un gate basado en rollouts (G0,G2,G3,G4,G8)."""
    cells = gc.cells_for_gate(gate_id, student_kind=student_kind)
    per_cell = {}
    verify_all_ok = True
    verify_detail = []
    raw_dir = out_dir / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for i, cell in enumerate(cells):
        res = run_rollouts(cell.config, seeds=seeds, swaps=cell.swaps,
                           test_agent_key=cell.test_agent_key)
        # verificacion (recomputo del score desde el log crudo)
        ok, det = verify_rollouts(res, tol=0.0)
        verify_all_ok = verify_all_ok and ok
        verify_detail.extend(det)
        # agregacion + guardar log crudo (por celda)
        steps_all = [ep["steps"] for ep in res["episodes"]]
        per_cell[cell.label] = gc.aggregate_cell(res["episodes"], steps_all)
        # step logs crudos a JSONL (sin duplicar en results.json)
        with open(raw_dir / f"cell{i:02d}.jsonl", "w") as fh:
            for ep in res["episodes"]:
                fh.write(json.dumps({
                    "label": cell.label, "seed": ep["seed"], "role_swap": ep["role_swap"],
                    "score": ep["score"], "soups": ep["soups"], "timeouts": ep["timeouts"],
                    "steps": ep["steps"],
                }) + "\n")
        # guardar el config exacto de la celda
        (out_dir / f"config_cell{i:02d}.json").write_text(json.dumps(cell.config, indent=2))

    return per_cell, verify_all_ok, verify_detail


def run_gate(gate_id: str, student_kind: str = "planner") -> bool:
    seeds = _load_seeds()
    out_dir = GATES_OUT / f"{gate_id}_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 0. Freeze check anti-trampa (§10.1) ---
    fok, fdet = verify_freeze()
    if not fok:
        print(f"[run_gate] ABORT: evaluation/ cambio tras el freeze: {fdet}")
        (out_dir / "results.json").write_text(json.dumps(
            {"gate": gate_id, "passed": False, "aborted": "freeze_mismatch", "detail": fdet}, indent=2))
        return False

    results: dict = {"gate": gate_id, "student_kind": student_kind, "timestamp": _timestamp()}
    passed = False
    metric: dict = {}

    if gate_id == "G1":
        # 1a. score sintetico reproducible a mano
        syn_ok, syn_det = check_synthetic(DEFAULT_DELIVERY_REWARD)
        # 1b. verify == reporte en un rollout real (planner vs greedy)
        per_cell, vok, vdet = _run_rollout_gate("G1", "planner", seeds, out_dir)
        passed = bool(syn_ok and vok)
        metric = {"synthetic_ok": syn_ok, "verify_ok": vok, "per_cell": per_cell}
        results.update({"synthetic": syn_det, "verify": vdet, "per_cell": per_cell})
        if passed:
            sums = write_freeze()
            results["freeze"] = sums
            print(f"[run_gate] G1 PASS -> freeze de evaluation/ escrito ({len(sums)} archivos).")

    elif gate_id in ("G0", "G2", "G3", "G4"):
        per_cell, vok, vdet = _run_rollout_gate(gate_id, student_kind, seeds, out_dir)
        if not vok:
            print("[run_gate] ABORT: verify.py != score reportado (posible bug/trampa).")
            passed = False
        else:
            passed, detail = gc.criterion(gate_id, per_cell)
            metric = {"per_cell": detail, "verify_ok": vok}
            results["per_cell"] = detail
        results["verify"] = vdet

    elif gate_id == "G8":
        passed, metric, results = _run_g8(seeds, out_dir, results)

    elif gate_id == "G5":
        passed, metric = _run_g5(out_dir)
        results["g5"] = metric

    elif gate_id in ("G6", "G7"):
        print(f"[run_gate] {gate_id} requiere un modelo PPO entrenado (FASE 3). "
              f"Usar: sbatch sbatch/train/... y luego este gate con --model.")
        results["note"] = "requiere modelo PPO (FASE 3)"
        passed = False
    else:
        raise SystemExit(f"gate desconocido: {gate_id}")

    results["passed"] = passed
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    _write_progress(gate_id, passed, metric, str(out_dir.relative_to(REPO)))

    print(f"\n[run_gate] {gate_id}: {'PASS' if passed else 'FAIL'}")
    print(f"[run_gate] artefactos: {out_dir}")
    print(json.dumps(metric, indent=2, default=str)[:2000])
    return passed


def _run_g8(seeds, out_dir, results):
    """G8: el selector nunca empeora vs planner puro y es robusto (0 crashes/timeouts)."""
    # baseline planner puro por celda (mismos seeds/compañeros/swap)
    student_cells = gc.cells_for_gate("G8", student_kind="student")
    per_cell = {}
    passed = True
    verify_all_ok = True
    raw_dir = out_dir / "raw_logs"; raw_dir.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(student_cells):
        res_s = run_rollouts(cell.config, seeds=seeds, swaps=cell.swaps, test_agent_key="agent_0")
        ok_s, _ = verify_rollouts(res_s, tol=0.0)
        # baseline: mismo config pero agent_0 = planner puro
        base_cfg = json.loads(json.dumps(cell.config))
        base_cfg["policies"]["agent_0"] = gc.planner_agent()
        res_p = run_rollouts(base_cfg, seeds=seeds, swaps=cell.swaps, test_agent_key="agent_0")
        verify_all_ok = verify_all_ok and ok_s
        agg_s = gc.aggregate_cell(res_s["episodes"], [e["steps"] for e in res_s["episodes"]])
        agg_p = gc.aggregate_cell(res_p["episodes"], [e["steps"] for e in res_p["episodes"]])
        cell_ok = (agg_s["timeouts"] == 0 and agg_s["invalids"] == 0
                   and agg_s["mean_score"] >= agg_p["mean_score"] - 1e-6)
        passed = passed and cell_ok
        per_cell[cell.label] = {"ok": cell_ok, "student": agg_s, "planner_baseline": agg_p}
        with open(raw_dir / f"g8_cell{i:02d}.jsonl", "w") as fh:
            for ep in res_s["episodes"]:
                fh.write(json.dumps({"label": cell.label, **{k: ep[k] for k in
                         ("seed", "role_swap", "score", "soups", "timeouts")}}) + "\n")
    passed = passed and verify_all_ok
    results["per_cell"] = per_cell
    return passed, {"per_cell": per_cell, "verify_ok": verify_all_ok}, results


def _run_g5(out_dir):
    """G5: gym env valido (check_env) + PPO 50k sin NaN y shaped sube vs aleatorio."""
    try:
        from training.gym_env import make_check_env_report
        report = make_check_env_report()
        (out_dir / "g5_report.json").write_text(json.dumps(report, indent=2, default=str))
        passed = bool(report.get("check_env_ok") and report.get("no_nan")
                      and report.get("shaped_improves"))
        return passed, report
    except Exception as exc:
        return False, {"error": repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, help="G0..G8")
    ap.add_argument("--student", default="planner",
                    choices=["planner", "student", "template_stay"],
                    help="que agente ocupa agent_0 en gates de rollout")
    args = ap.parse_args()
    ok = run_gate(args.gate, student_kind=args.student)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
