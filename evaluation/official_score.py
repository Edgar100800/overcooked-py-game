"""Score oficial de la competencia (PLAN.md §1, §6).

Formula EXACTA:
    score = 10000*sopas
            + 10*(horizon - t_ultima_sopa)
            + (horizon - t_primera_sopa)
            - min(100*timeouts, 5000)
Si sopas == 0 -> score = 0.

La VERDAD son los eventos del entorno (PLAN §10.3): las sopas y los tiempos se
computan desde el sparse reward por step (>0 == entrega de sopa) registrado en el
step log crudo, NUNCA desde contadores del agente. `verify.py` recomputa con estas
mismas funciones y debe coincidir con lo reportado (tolerancia 0).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# Valor por defecto de la recompensa por entrega en overcooked_ai (se puede
# sobreescribir en runtime con el valor real del mdp: `mdp.delivery_reward`).
DEFAULT_DELIVERY_REWARD = 20.0


@dataclass
class ScoreBreakdown:
    score: float
    soups: int
    t_first: int | None
    t_last: int | None
    timeouts: int
    horizon: int
    penalty: float
    total_sparse: float

    def to_dict(self) -> dict:
        return asdict(self)


def count_soups(sparse_per_step, delivery_reward: float = DEFAULT_DELIVERY_REWARD) -> int:
    """Numero de sopas entregadas = sparse total / recompensa por entrega.

    Robusto a entregas simultaneas de ambos agentes en el mismo step (sparse=2*R).
    Si `delivery_reward` no es valido, cae a contar steps con sparse>0.
    """
    total = float(sum(sparse_per_step))
    if delivery_reward and delivery_reward > 0:
        return int(round(total / delivery_reward))
    return int(sum(1 for r in sparse_per_step if r > 0))


def delivery_timesteps(sparse_per_step, timesteps=None) -> list[int]:
    """Timesteps (segun el env) en los que hubo al menos una entrega (sparse>0)."""
    out = []
    for i, r in enumerate(sparse_per_step):
        if r > 0:
            out.append(int(timesteps[i]) if timesteps is not None else i)
    return out


def official_score(
    sparse_per_step,
    timeouts: int,
    horizon: int,
    delivery_reward: float = DEFAULT_DELIVERY_REWARD,
    timesteps=None,
) -> ScoreBreakdown:
    """Computa el score oficial de UN rollout.

    Args:
        sparse_per_step: lista (len T) del sparse reward del env en cada step (>=0).
        timeouts: nº de timeouts del SafeActionWrapper en el rollout.
        horizon: horizonte del episodio.
        delivery_reward: recompensa por sopa (default 20; usar mdp.delivery_reward).
        timesteps: timesteps del env alineados con sparse_per_step (opcional).
    """
    soups = count_soups(sparse_per_step, delivery_reward)
    total_sparse = float(sum(sparse_per_step))
    penalty = float(min(100 * int(timeouts), 5000))

    if soups <= 0:
        return ScoreBreakdown(
            score=0.0, soups=0, t_first=None, t_last=None,
            timeouts=int(timeouts), horizon=int(horizon),
            penalty=penalty, total_sparse=total_sparse,
        )

    deliv = delivery_timesteps(sparse_per_step, timesteps)
    t_first = deliv[0]
    t_last = deliv[-1]
    score = (
        10000.0 * soups
        + 10.0 * (horizon - t_last)
        + (horizon - t_first)
        - penalty
    )
    return ScoreBreakdown(
        score=float(score), soups=int(soups), t_first=int(t_first), t_last=int(t_last),
        timeouts=int(timeouts), horizon=int(horizon),
        penalty=penalty, total_sparse=total_sparse,
    )


def score_from_step_log(steps: list[dict], horizon: int, timeouts: int,
                        delivery_reward: float = DEFAULT_DELIVERY_REWARD) -> ScoreBreakdown:
    """Computa el score desde el step log crudo (lista de dicts por step).

    Cada dict debe tener 'sparse' (float, sparse reward del env en ese step) y
    'timestep' (int, timestep del env). Usado por run_gate.py y verify.py.
    """
    sparse_per_step = [float(s.get("sparse", 0.0)) for s in steps]
    timesteps = [int(s.get("timestep", i)) for i, s in enumerate(steps)]
    return official_score(sparse_per_step, timeouts, horizon, delivery_reward, timesteps)


# --- Casos sinteticos para G1 (score reproducible a mano) --------------------
def synthetic_cases(delivery_reward: float = DEFAULT_DELIVERY_REWARD) -> list[dict]:
    """3 casos sinteticos con score esperado calculado a mano (PLAN §11 G1)."""
    H = 250
    R = delivery_reward
    cases = []

    # Caso 1: 0 sopas -> score 0.
    cases.append({
        "name": "cero_sopas",
        "sparse_per_step": [0.0] * H,
        "timeouts": 0, "horizon": H, "expected": 0.0,
    })

    # Caso 2: 1 sopa en t=100, con 3 timeouts.
    sp = [0.0] * H
    sp[100] = R
    expected2 = 10000 + 10 * (H - 100) + (H - 100) - min(100 * 3, 5000)
    cases.append({
        "name": "una_sopa_con_timeouts",
        "sparse_per_step": sp,
        "timeouts": 3, "horizon": H, "expected": float(expected2),
    })

    # Caso 3: 2 sopas (t=50 y t=200), 0 timeouts.
    sp = [0.0] * H
    sp[50] = R
    sp[200] = R
    expected3 = 10000 * 2 + 10 * (H - 200) + (H - 50) - 0
    cases.append({
        "name": "dos_sopas_normal",
        "sparse_per_step": sp,
        "timeouts": 0, "horizon": H, "expected": float(expected3),
    })
    return cases


def check_synthetic(delivery_reward: float = DEFAULT_DELIVERY_REWARD) -> tuple[bool, list[dict]]:
    """Valida los casos sinteticos. Devuelve (todos_ok, detalle)."""
    detail = []
    all_ok = True
    for c in synthetic_cases(delivery_reward):
        sb = official_score(c["sparse_per_step"], c["timeouts"], c["horizon"], delivery_reward)
        ok = abs(sb.score - c["expected"]) < 1e-6
        all_ok = all_ok and ok
        detail.append({"name": c["name"], "expected": c["expected"], "got": sb.score, "ok": ok})
    return all_ok, detail


if __name__ == "__main__":
    import json
    ok, detail = check_synthetic()
    print(json.dumps({"all_ok": ok, "cases": detail}, indent=2))
