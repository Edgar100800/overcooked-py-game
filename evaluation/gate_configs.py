"""Definicion de gates y construccion de configs (PLAN.md §11).

Se separa de run_gate.py para que la logica de orquestacion quede limpia. NOTA:
gate_configs.py NO esta en la lista de archivos congelados (FROZEN_FILES en
verify.py); los archivos congelados son official_score/run_gate/verify/gate_seeds.
Los criterios numericos de aprobacion viven aqui pero se ejecutan desde run_gate.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Layouts: 3 oficiales + 3 custom del repo (PLAN §11 G3/G4).
OFFICIAL_LAYOUTS = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
CUSTOM_LAYOUTS = [
    ("custom_room", "configs/layouts/custom_room.layout"),
    ("custom_dual_pots", "configs/layouts/custom_dual_pots.layout"),
    ("custom_zigzag_kitchen", "configs/layouts/custom_zigzag_kitchen.layout"),
]
ALL_SIX = [(name, None) for name in OFFICIAL_LAYOUTS] + CUSTOM_LAYOUTS

_WRAP = {"random_action_prob": 0.0, "max_action_time_ms": 100,
         "invalid_action": "stay", "timeout_action": "stay"}


# ------------------------------------------------------------ agentes/partners
def partner(kind: str) -> dict:
    if kind == "stay":
        return {"type": "builtin", "name": "stay", **_WRAP}
    if kind == "random_motion":
        return {"type": "builtin", "name": "random_motion", **_WRAP}
    if kind == "greedy":
        return {"type": "builtin", "name": "greedy_full_task",
                "ingredient": "onion", "avoid_teammate": True, **_WRAP}
    if kind == "greedy_eps":
        c = {"type": "builtin", "name": "greedy_full_task",
             "ingredient": "onion", "avoid_teammate": True, **_WRAP}
        c["random_action_prob"] = 0.15
        return c
    raise ValueError(f"partner desconocido: {kind}")


def planner_agent(config: dict | None = None) -> dict:
    return {"type": "python_class", "path": "policies/planner_agent.py",
            "class_name": "PlannerAgent", "name": "planner",
            "config": config or {"ingredient": "onion"}, **_WRAP}


def student_agent(config: dict | None = None) -> dict:
    return {"type": "python_class", "path": "policies/student_agent.py",
            "class_name": "StudentAgent", "name": "student",
            "config": config or {}, **_WRAP}


def template_stay() -> dict:
    return {"type": "python_class", "path": "policies/template.py",
            "class_name": "StudentAgent", "name": "template_stay",
            "config": {"action": "stay"}, **_WRAP}


def make_config(layout: str, layout_file: str | None, agent0: dict, agent1: dict,
                horizon: int = 250, obs_type: str = "state") -> dict:
    return {
        "seed": 0,
        "environment": {
            "layout_name": None if layout_file else layout,
            "layout_file": layout_file,
            "horizon": horizon,
            "old_dynamics": True,
        },
        "policies": {"agent_0": agent0, "agent_1": agent1},
        "observation": {"type": obs_type, "include_agent_index": True},
        "rendering": {"mode": "none"},
        "logging": {"save_step_log": False, "save_episode_summary": False,
                    "output_dir": "outputs/_gate_tmp"},
    }


# --------------------------------------------------------------------- celdas
@dataclass
class Cell:
    layout: str
    config: dict
    swaps: list          # [False] o [False, True]
    label: str
    test_agent_key: str = "agent_0"


def cells_for_gate(gate_id: str, student_kind: str = "planner") -> list[Cell]:
    """Construye las celdas de rollout de cada gate.

    student_kind: 'planner' (G2-G4) o 'student' (G8) o 'template_stay' (G0).
    """
    def a0():
        return {"planner": planner_agent, "student": student_agent,
                "template_stay": template_stay}[student_kind]()

    cells = []
    if gate_id == "G0":
        # Smoke: greedy (player 0) entrega solo -> prueba la deteccion de entregas.
        # greedy va en agent_0 y el template-stay en agent_1 (greedy-como-player-1
        # se bloquearia con el dispensador de platos en cramped_room).
        cfg = make_config("cramped_room", None, partner("greedy"), template_stay())
        cells.append(Cell("cramped_room", cfg, [False], "G0/cramped_room/greedy-vs-stay",
                          test_agent_key="agent_0"))
    elif gate_id == "G1":
        cfg = make_config("cramped_room", None, planner_agent(), partner("greedy"))
        cells.append(Cell("cramped_room", cfg, [False], "G1/cramped_room/planner-vs-greedy"))
    elif gate_id == "G2":
        cfg = make_config("cramped_room", None, a0(), partner("stay"))
        cells.append(Cell("cramped_room", cfg, [False, True], "G2/cramped_room/planner-vs-stay"))
    elif gate_id == "G3":
        for name, lf in ALL_SIX:
            cfg = make_config(name, lf, a0(), partner("greedy"))
            cells.append(Cell(name, cfg, [False, True], f"G3/{name}/planner-vs-greedy"))
    elif gate_id == "G4":
        for name, lf in ALL_SIX:
            cfg = make_config(name, lf, a0(), partner("random_motion"))
            cells.append(Cell(name, cfg, [False, True], f"G4/{name}/planner-vs-random"))
    elif gate_id == "G8":
        for name, lf in ALL_SIX:
            scfg = {"layout": name, "layout_file": lf}  # activa PPO si esta habilitado
            for pk in ("greedy", "greedy_eps", "random_motion"):
                cfg = make_config(name, lf, student_agent(scfg), partner(pk))
                cells.append(Cell(name, cfg, [False, True], f"G8/{name}/student-vs-{pk}"))
    else:
        raise ValueError(f"gate {gate_id} no tiene celdas de rollout (ver run_gate.py)")
    return cells


# ------------------------------------------------------------------- criterios
def aggregate_cell(episodes: list[dict], steps_all: list[list[dict]]) -> dict:
    import numpy as np
    soups = [e["soups"] for e in episodes]
    scores = [e["score"] for e in episodes]
    timeouts = sum(e["timeouts"] for e in episodes)
    invalids = sum(e.get("invalids", 0) for e in episodes)
    lat = [s["our_elapsed_ms"] for steps in steps_all for s in steps
           if s.get("our_elapsed_ms") is not None]
    lat = np.array(lat) if lat else np.array([0.0])
    return {
        "mean_soups": float(np.mean(soups)) if soups else 0.0,
        "min_soups": float(np.min(soups)) if soups else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "timeouts": int(timeouts),
        "invalids": int(invalids),
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p99": float(np.percentile(lat, 99)),
        "latency_max": float(lat.max()),
        "n_episodes": len(episodes),
    }


def criterion(gate_id: str, per_cell: dict[str, dict]) -> tuple[bool, dict]:
    """Aplica el criterio numerico del gate (PLAN §11). Devuelve (passed, resumen)."""
    detail = {}
    passed = True
    for label, agg in per_cell.items():
        if gate_id == "G0":
            ok = agg["mean_soups"] >= 1.0
        elif gate_id == "G2":
            ok = agg["mean_soups"] >= 1.0 and agg["timeouts"] == 0 and agg["latency_p99"] < 50.0
        elif gate_id == "G3":
            ok = agg["mean_soups"] >= 2.0 and agg["timeouts"] == 0
        elif gate_id == "G4":
            ok = agg["mean_soups"] >= 1.0
        elif gate_id == "G8":
            # comparacion vs planner puro se maneja en run_gate (necesita baseline);
            # aqui solo el criterio de robustez.
            ok = agg["timeouts"] == 0 and agg["invalids"] == 0
        else:
            ok = True
        detail[label] = {"ok": ok, **agg}
        passed = passed and ok
    return passed, detail
