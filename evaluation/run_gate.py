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


def _run_g6_g7(gate_id, layout, layout_file, model_path, seeds, out_dir, results):
    """G6: PPO vs greedy -> score>0 (>=1 sopa). G7: PPO > planner (con/sin swap) ->
    habilita el modelo (models/<key>/enabled). Requiere models/<key>/best.zip."""
    import numpy as np
    key = Path(layout_file).stem if layout_file else layout
    model = model_path or str(REPO / "models" / key / "best.zip")
    if not Path(model).exists():
        print(f"[run_gate] {gate_id}: falta el modelo {model}. Entrena primero "
              f"(sbatch sbatch/train/run_train_ppo.sh).")
        return False, {"error": f"modelo no encontrado: {model}"}, results

    scfg = {"layout": layout, "layout_file": layout_file,
            "model_path": model, "require_enabled": False}
    student = gc.student_agent(scfg)
    cfg_student = gc.make_config(layout, layout_file, student, gc.partner("greedy"))
    swaps = [False, True]
    res_s = run_rollouts(cfg_student, seeds=seeds, swaps=swaps, test_agent_key="agent_0")
    ok_s, _ = verify_rollouts(res_s, tol=0.0)
    agg_s = gc.aggregate_cell(res_s["episodes"], [e["steps"] for e in res_s["episodes"]])

    if gate_id == "G6":
        passed = bool(ok_s and agg_s["mean_soups"] >= 1.0)
        metric = {"student": agg_s, "verify_ok": ok_s}
        results["per_cell"] = {f"G6/{key}/ppo-vs-greedy": {"ok": passed, **agg_s}}
        return passed, metric, results

    # G7: comparar vs planner puro, por rol (swap) individual.
    cfg_planner = gc.make_config(layout, layout_file, gc.planner_agent(), gc.partner("greedy"))
    res_p = run_rollouts(cfg_planner, seeds=seeds, swaps=swaps, test_agent_key="agent_0")

    def mean_score_by_swap(res, swap):
        v = [e["score"] for e in res["episodes"] if e["role_swap"] == swap]
        return float(np.mean(v)) if v else 0.0

    beats = all(mean_score_by_swap(res_s, sw) > mean_score_by_swap(res_p, sw) for sw in swaps)
    latency_ok = agg_s["latency_p99"] < 50.0
    passed = bool(ok_s and beats and latency_ok)

    detail = {sw and "swap" or "noswap":
              {"ppo": mean_score_by_swap(res_s, sw), "planner": mean_score_by_swap(res_p, sw)}
              for sw in swaps}
    metric = {"beats_planner": beats, "latency_p99": agg_s["latency_p99"],
              "latency_ok": latency_ok, "by_swap": detail, "verify_ok": ok_s}
    results["per_cell"] = {f"G7/{key}": {"ok": passed, **metric}}

    if passed:
        enabled = REPO / "models" / key / "enabled"
        enabled.parent.mkdir(parents=True, exist_ok=True)
        enabled.write_text(f"habilitado por G7: PPO>planner en gate_seeds\n{detail}\n")
        print(f"[run_gate] G7 PASS -> modelo habilitado: {enabled}")
    else:
        print(f"[run_gate] G7: PPO NO supera al planner en {key} -> selector queda en "
              f"planner (decision valida, PLAN §11 G7).")
    return passed, metric, results


def run_gate(gate_id: str, student_kind: str = "planner",
             layout: str = "cramped_room", layout_file: str | None = None,
             model_path: str | None = None) -> bool:
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
        passed, metric, results = _run_g6_g7(gate_id, layout, layout_file, model_path,
                                             seeds, out_dir, results)
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
    ap.add_argument("--layout", default="cramped_room", help="layout para G6/G7")
    ap.add_argument("--layout-file", default=None, help=".layout custom para G6/G7")
    ap.add_argument("--model", default=None, help="ruta a best.zip (G6/G7)")
    args = ap.parse_args()
    ok = run_gate(args.gate, student_kind=args.student, layout=args.layout,
                  layout_file=args.layout_file, model_path=args.model)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
