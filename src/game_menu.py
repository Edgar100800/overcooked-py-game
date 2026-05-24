"""Interactive launcher for collecting Overcooked demonstrations.

Usage:
    python -m src.game_menu
"""

from __future__ import annotations

import copy
import os
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from src.config import load_yaml
from src.dataset_progress import (
    RECORDINGS_PER_SCENARIO,
    TARGET_SCENARIOS,
    TARGET_RECORDINGS,
    build_progress_report,
    collect_recordings,
)


BASE_CONFIG = Path("configs/collect_demonstrations.yaml")
DATA_DIR = Path("data/demonstrations")

class RecordingPlanItem(NamedTuple):
    layout_name: str
    agent_name: str
    layout_file: str | None = None

    @property
    def progress_name(self) -> str:
        if self.layout_file:
            return Path(self.layout_file).stem
        return self.layout_name

    @property
    def display_name(self) -> str:
        if self.layout_file:
            return f"{self.progress_name} (custom)"
        return self.layout_name


AGENTS = ["greedy_full_task", "random_motion", "stay"]
LEVELS = [
    RecordingPlanItem("asymmetric_advantages", "greedy_full_task"),
    RecordingPlanItem("coordination_ring", "greedy_full_task"),
    RecordingPlanItem("counter_circuit", "greedy_full_task"),
    RecordingPlanItem("cramped_room", "greedy_full_task"),
    RecordingPlanItem("forced_coordination", "greedy_full_task"),
    RecordingPlanItem("large_room", "greedy_full_task"),
    RecordingPlanItem("simple_o", "greedy_full_task"),
    RecordingPlanItem("simple_tomato", "greedy_full_task"),
    RecordingPlanItem("small_corridor", "greedy_full_task"),
    RecordingPlanItem("soup_coordination", "greedy_full_task"),
    RecordingPlanItem("custom_zigzag_kitchen", "greedy_full_task", "configs/layouts/custom_zigzag_kitchen.layout"),
    RecordingPlanItem("custom_dual_pots", "greedy_full_task", "configs/layouts/custom_dual_pots.layout"),
]


RECORDING_PLAN = [
    RecordingPlanItem("cramped_room", "greedy_full_task"),
    RecordingPlanItem("cramped_room", "random_motion"),
    RecordingPlanItem("cramped_room", "stay"),
    RecordingPlanItem("asymmetric_advantages", "stay"),
    RecordingPlanItem("asymmetric_advantages", "greedy_full_task"),
    RecordingPlanItem("asymmetric_advantages", "random_motion"),
    RecordingPlanItem("coordination_ring", "random_motion"),
    RecordingPlanItem("coordination_ring", "stay"),
    RecordingPlanItem("coordination_ring", "greedy_full_task"),
    RecordingPlanItem("counter_circuit", "greedy_full_task"),
    RecordingPlanItem("counter_circuit", "random_motion"),
    RecordingPlanItem("counter_circuit", "stay"),
    RecordingPlanItem("forced_coordination", "stay"),
    RecordingPlanItem("forced_coordination", "greedy_full_task"),
    RecordingPlanItem("forced_coordination", "random_motion"),
    RecordingPlanItem("large_room", "random_motion"),
    RecordingPlanItem("large_room", "stay"),
    RecordingPlanItem("large_room", "greedy_full_task"),
    RecordingPlanItem("simple_o", "greedy_full_task"),
    RecordingPlanItem("simple_o", "random_motion"),
    RecordingPlanItem("simple_o", "stay"),
    RecordingPlanItem("simple_tomato", "stay"),
    RecordingPlanItem("simple_tomato", "greedy_full_task"),
    RecordingPlanItem("simple_tomato", "random_motion"),
    RecordingPlanItem("small_corridor", "random_motion"),
    RecordingPlanItem("small_corridor", "stay"),
    RecordingPlanItem("small_corridor", "greedy_full_task"),
    RecordingPlanItem("soup_coordination", "greedy_full_task"),
    RecordingPlanItem("soup_coordination", "random_motion"),
    RecordingPlanItem("soup_coordination", "stay"),
    RecordingPlanItem("custom_zigzag_kitchen", "stay", "configs/layouts/custom_zigzag_kitchen.layout"),
    RecordingPlanItem("custom_zigzag_kitchen", "greedy_full_task", "configs/layouts/custom_zigzag_kitchen.layout"),
    RecordingPlanItem("custom_zigzag_kitchen", "random_motion", "configs/layouts/custom_zigzag_kitchen.layout"),
    RecordingPlanItem("custom_dual_pots", "random_motion", "configs/layouts/custom_dual_pots.layout"),
    RecordingPlanItem("custom_dual_pots", "stay", "configs/layouts/custom_dual_pots.layout"),
    RecordingPlanItem("custom_dual_pots", "greedy_full_task", "configs/layouts/custom_dual_pots.layout"),
]


def main() -> None:
    while True:
        _clear_screen()
        _print_header()
        _print_compact_progress()
        print()
        print("1. Jugar siguiente grabacion recomendada")
        print("2. Elegir nivel y agente manualmente")
        print("3. Ver progreso detallado")
        print("4. Como se juega")
        print("5. Salir")
        choice = input("\nElige una opcion: ").strip()

        if choice == "1":
            item = _next_plan_item()
            if item is None:
                _pause("Ya tienes el plan completo. Revisa el progreso detallado antes de entregar.")
                continue
            _run_recording(item)
        elif choice == "2":
            item = _choose_manual_item()
            if item is not None:
                _run_recording(item)
        elif choice == "3":
            _clear_screen()
            print(build_progress_report())
            _pause()
        elif choice == "4":
            _clear_screen()
            _print_how_to_play()
            _pause()
        elif choice == "5":
            print("Saliendo.")
            return
        else:
            _pause("Opcion no valida.")


def _print_header() -> None:
    print("====================================")
    print(" Overcooked - Menu de grabaciones")
    print("====================================")


def _print_compact_progress() -> None:
    recordings = [r for r in collect_recordings(DATA_DIR) if r.complete]
    scenario_counts = Counter(r.layout_name for r in recordings)
    complete_scenarios = sum(1 for count in scenario_counts.values() if count >= RECORDINGS_PER_SCENARIO)
    next_item = _next_plan_item()

    print(f"Progreso general: {len(recordings)}/{TARGET_RECORDINGS} grabaciones completas")
    print(f"Escenarios listos: {complete_scenarios}/{TARGET_SCENARIOS}")
    if next_item is None:
        print("Siguiente: plan completo")
    else:
        print(f"Siguiente: nivel={next_item.display_name} | agente automatico={next_item.agent_name}")


def _next_plan_item() -> RecordingPlanItem | None:
    recordings = [r for r in collect_recordings(DATA_DIR) if r.complete]
    scenario_counts = Counter(r.layout_name for r in recordings)
    required_counts: Counter[str] = Counter()

    for item in RECORDING_PLAN:
        required_counts[item.progress_name] += 1
        if scenario_counts[item.progress_name] < required_counts[item.progress_name]:
            return item
    return None


def _choose_manual_item() -> RecordingPlanItem | None:
    _clear_screen()
    print("Elige nivel:")
    for idx, level in enumerate(LEVELS, start=1):
        print(f"{idx:2d}. {level.display_name}")
    print(" 0. Cancelar")

    layout_idx = _read_int("\nNivel: ", minimum=0, maximum=len(LEVELS))
    if layout_idx == 0:
        return None
    level = LEVELS[layout_idx - 1]

    print("\nElige agente automatico:")
    for idx, agent_name in enumerate(AGENTS, start=1):
        print(f"{idx}. {agent_name}")
    agent_idx = _read_int("\nAgente: ", minimum=1, maximum=len(AGENTS))
    return RecordingPlanItem(
        layout_name=level.layout_name,
        agent_name=AGENTS[agent_idx - 1],
        layout_file=level.layout_file,
    )


def _run_recording(item: RecordingPlanItem) -> None:
    _clear_screen()
    recordings_before = len([r for r in collect_recordings(DATA_DIR) if r.complete])
    recording_number = min(recordings_before + 1, TARGET_RECORDINGS)

    print("Preparando grabacion")
    print(f"Grabacion: {recording_number}/{TARGET_RECORDINGS}")
    print(f"Nivel: {item.display_name}")
    print(f"Agente automatico: {item.agent_name}")
    print()
    _print_how_to_play(short=True)
    input("\nPresiona Enter para abrir el juego...")

    from src.runner import run_from_config

    config = _config_for_item(item, recording_number=recording_number)
    result = run_from_config(config)

    print()
    print("Resultado de la partida:")
    print(f"  - Rollouts: {result.get('num_rollouts')}")
    print(f"  - Reward promedio: {result.get('mean_return_sparse')}")
    print()
    print(build_progress_report())
    _pause()


def _config_for_item(item: RecordingPlanItem, recording_number: int) -> dict:
    config = copy.deepcopy(load_yaml(BASE_CONFIG))
    config.setdefault("environment", {})
    config["environment"]["layout_name"] = item.layout_name
    config["environment"]["layout_file"] = item.layout_file

    config.setdefault("policies", {}).setdefault("agent_0", {})
    config["policies"]["agent_0"]["type"] = "builtin"
    config["policies"]["agent_0"]["name"] = item.agent_name

    config.setdefault("rendering", {})
    config["rendering"]["mode"] = "window"
    config["rendering"]["window_scale"] = 2.0
    config["rendering"]["window_caption"] = (
        f"Overcooked AI | Grabacion {recording_number}/{TARGET_RECORDINGS}"
    )

    return config


def _print_how_to_play(short: bool = False) -> None:
    print("Como se juega")
    print("------------")
    print("Objetivo: cooperar con el agente automatico para preparar y entregar sopa.")
    print("Tu controlas al agente humano. La partida dura 250 pasos.")
    print()
    print("Controles:")
    print("  - Moverse: flechas o W/A/S/D")
    print("  - Interactuar/tomar/dejar/servir: Space, E o Enter")
    print("  - Salir/cancelar partida: Escape o Q")
    print()
    print("Durante el juego:")
    print("  - El titulo de la ventana muestra el nivel y el paso actual.")
    print("  - Cuando termina la partida, el menu muestra el progreso actualizado.")
    if not short:
        print()
        print("Flujo recomendado:")
        print("  1. Entra a 'Jugar siguiente grabacion recomendada'.")
        print("  2. Completa los 250 pasos sin cerrar la ventana.")
        print("  3. Revisa el progreso.")
        print(f"  4. Repite hasta llegar a {TARGET_RECORDINGS}/{TARGET_RECORDINGS} grabaciones y {TARGET_SCENARIOS}/{TARGET_SCENARIOS} escenarios.")


def _read_int(prompt: str, *, minimum: int, maximum: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Ingresa un numero valido.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Ingresa un numero entre {minimum} y {maximum}.")


def _pause(message: str | None = None) -> None:
    if message:
        print(message)
    input("\nPresiona Enter para continuar...")


def _clear_screen() -> None:
    command = "cls" if os.name == "nt" else "clear"
    os.system(command)


if __name__ == "__main__":
    main()
