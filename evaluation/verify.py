"""verify.py — recomputa el score desde el step log crudo (PLAN.md §10.3).

La verdad son los eventos del entorno. Este modulo recalcula el score oficial de
cada episodio a partir del step log crudo (sparse por step + timeouts) y lo compara
con el score reportado por run_gate.py. Deben coincidir con tolerancia 0.

Tambien provee `compute_checksums`/`verify_freeze` para el congelamiento anti-trampa
de evaluation/ (§10.1).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.official_score import score_from_step_log

EVAL_DIR = Path(__file__).resolve().parent
# Archivos congelados al pasar G1 (§10.1).
FROZEN_FILES = ["official_score.py", "run_gate.py", "verify.py", "gate_seeds.json"]
FREEZE_PATH = EVAL_DIR / "freeze.sha256"


# ------------------------------------------------------- re-computo del score
def verify_episode(ep: dict, delivery_reward: float, tol: float = 0.0) -> dict:
    """Recomputa el score de un episodio desde ep['steps'] y compara con ep['score']."""
    steps = ep.get("steps", [])
    horizon = int(ep.get("horizon", steps[-1]["timestep"] + 1 if steps else 0))
    timeouts = int(ep.get("timeouts", 0))
    recomputed = score_from_step_log(steps, horizon, timeouts, delivery_reward)
    reported = float(ep.get("score", 0.0))
    diff = abs(recomputed.score - reported)
    return {
        "seed": ep.get("seed"),
        "role_swap": ep.get("role_swap"),
        "reported_score": reported,
        "recomputed_score": recomputed.score,
        "diff": diff,
        "ok": diff <= tol,
    }


def verify_rollouts(rollout_result: dict, tol: float = 0.0) -> tuple[bool, list[dict]]:
    """Verifica todos los episodios de un run_rollouts(). Devuelve (todos_ok, detalle)."""
    delivery_reward = float(rollout_result.get("delivery_reward", 20.0))
    horizon = int(rollout_result.get("horizon", 0))
    detail = []
    all_ok = True
    for ep in rollout_result.get("episodes", []):
        ep = {**ep, "horizon": ep.get("horizon", horizon)}
        r = verify_episode(ep, delivery_reward, tol)
        all_ok = all_ok and r["ok"]
        detail.append(r)
    return all_ok, detail


# ------------------------------------------------------------------- freeze
def compute_checksums() -> dict[str, str]:
    out = {}
    for name in FROZEN_FILES:
        p = EVAL_DIR / name
        out[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
    return out


def write_freeze() -> dict[str, str]:
    sums = compute_checksums()
    FREEZE_PATH.write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n")
    return sums


def verify_freeze() -> tuple[bool, dict]:
    """Verifica que evaluation/ no cambio tras el freeze. (ok, detalle).

    Nota: run_gate.py se recomputa a si mismo; para evitar el problema del huevo y
    la gallina se excluye el propio run_gate.py del hash SOLO si aun no hay freeze.
    Con freeze presente, todos los archivos deben cuadrar exactamente.
    """
    if not FREEZE_PATH.exists():
        return True, {"frozen": False, "reason": "no freeze yet"}
    expected = json.loads(FREEZE_PATH.read_text())
    actual = compute_checksums()
    mismatches = {k: {"expected": expected.get(k), "actual": actual.get(k)}
                  for k in FROZEN_FILES if expected.get(k) != actual.get(k)}
    return (len(mismatches) == 0), {"frozen": True, "mismatches": mismatches}


if __name__ == "__main__":
    ok, det = verify_freeze()
    print(json.dumps({"freeze_ok": ok, "detail": det}, indent=2))
