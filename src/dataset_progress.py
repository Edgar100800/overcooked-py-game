"""Progress checker for the Overcooked demonstration dataset.

Usage:
    python -m src.dataset_progress
    python -m src.dataset_progress --data-dir data/demonstrations
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_RECORDINGS = 36
TARGET_SCENARIOS = 12
RECORDINGS_PER_SCENARIO = 3
TARGET_TRANSITIONS = 250


@dataclass
class RecordingInfo:
    path: Path
    layout_name: str
    agent_0_name: str
    transitions: int
    complete: bool
    layout_file: str | None


def _load_pickle(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _get_nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def inspect_recording(path: Path, target_transitions: int = TARGET_TRANSITIONS) -> RecordingInfo | None:
    payload = _load_pickle(path)
    if payload is None:
        return None

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    records = payload.get("records", [])
    transitions = len(records) if isinstance(records, list) else 0

    layout_name = _get_nested(metadata, ["layout", "layout_name"])
    if not layout_name:
        layout_name = _get_nested(metadata, ["environment", "layout_name"], "unknown_layout")

    layout_file = _get_nested(metadata, ["layout", "layout_file"])
    if not layout_file:
        layout_file = _get_nested(metadata, ["environment", "layout_file"])

    agent_0_name = _get_nested(metadata, ["policies", "agent_0", "name"], "unknown_agent")

    return RecordingInfo(
        path=path,
        layout_name=str(layout_name),
        agent_0_name=str(agent_0_name),
        transitions=transitions,
        complete=transitions >= target_transitions,
        layout_file=None if layout_file in ("", "None", None) else str(layout_file),
    )


def collect_recordings(data_dir: Path, target_transitions: int = TARGET_TRANSITIONS) -> list[RecordingInfo]:
    if not data_dir.exists():
        return []

    recordings: list[RecordingInfo] = []
    for path in sorted(data_dir.glob("*.pkl")):
        info = inspect_recording(path, target_transitions=target_transitions)
        if info is not None:
            recordings.append(info)
    return recordings


def _status(ok: bool) -> str:
    return "OK" if ok else "FALTA"


def build_progress_report(
    *,
    data_dir: Path = Path("data/demonstrations"),
    integrantes_path: Path = Path("integrantes.txt"),
    layouts_dir: Path = Path("configs/layouts"),
    target_recordings: int = TARGET_RECORDINGS,
    target_scenarios: int = TARGET_SCENARIOS,
    recordings_per_scenario: int = RECORDINGS_PER_SCENARIO,
    target_transitions: int = TARGET_TRANSITIONS,
) -> str:
    recordings = collect_recordings(data_dir, target_transitions=target_transitions)
    complete_recordings = [r for r in recordings if r.complete]
    incomplete_recordings = [r for r in recordings if not r.complete]

    scenario_counts: dict[str, int] = defaultdict(int)
    for recording in complete_recordings:
        scenario_counts[recording.layout_name] += 1

    complete_scenarios = {
        scenario: count
        for scenario, count in scenario_counts.items()
        if count >= recordings_per_scenario
    }

    agent_counts = Counter(r.agent_0_name for r in complete_recordings)
    custom_layout_files = sorted({r.layout_file for r in complete_recordings if r.layout_file})
    local_layout_files = sorted(layouts_dir.glob("*.layout")) if layouts_dir.exists() else []

    lines = []
    lines.append("=== Progreso Overcooked ===")
    lines.append(f"{_status(len(complete_recordings) >= target_recordings)} grabaciones completas: {len(complete_recordings)}/{target_recordings}")
    lines.append(
        f"{_status(len(complete_scenarios) >= target_scenarios)} "
        f"escenarios con {recordings_per_scenario} grabaciones: {len(complete_scenarios)}/{target_scenarios}"
    )
    lines.append(f"{_status(integrantes_path.exists())} integrantes.txt: {integrantes_path}")
    lines.append(f"Grabaciones detectadas en: {data_dir}")
    lines.append("")

    if scenario_counts:
        lines.append("Escenarios:")
        for scenario, count in sorted(scenario_counts.items()):
            missing = max(0, recordings_per_scenario - count)
            suffix = "completo" if missing == 0 else f"faltan {missing}"
            lines.append(f"  - {scenario}: {count}/{recordings_per_scenario} ({suffix})")
        lines.append("")
    else:
        lines.append("Escenarios: todavia no hay grabaciones completas.")
        lines.append("")

    if incomplete_recordings:
        lines.append("Grabaciones incompletas o cortadas:")
        for recording in incomplete_recordings:
            lines.append(f"  - {recording.path.name}: {recording.transitions}/{target_transitions} transiciones")
        lines.append("")

    missing_recordings = max(0, target_recordings - len(complete_recordings))
    missing_scenarios = max(0, target_scenarios - len(complete_scenarios))
    lines.append("Resumen de lo que falta:")
    lines.append(f"  - Grabaciones completas faltantes: {missing_recordings}")
    lines.append(f"  - Escenarios completos faltantes: {missing_scenarios}")
    if missing_recordings > 0:
        lines.append(f"  - Siguiente meta: juega hasta tener {target_recordings} archivos .pkl completos.")
    lines.append("")

    lines.append("Agente automatico usado en grabaciones completas:")
    if agent_counts:
        for agent_name, count in sorted(agent_counts.items()):
            lines.append(f"  - {agent_name}: {count}")
    else:
        lines.append("  - todavia no hay datos")
    lines.append("")

    lines.append("Layouts custom:")
    if custom_layout_files:
        for layout_file in custom_layout_files:
            exists = Path(layout_file).exists()
            lines.append(f"  - {_status(exists)} usado en metadata: {layout_file}")
    elif local_layout_files:
        lines.append("  - Hay archivos .layout locales, adjuntalos solo si los usaste:")
        for path in local_layout_files:
            lines.append(f"    {path}")
    else:
        lines.append("  - No se detectaron archivos .layout locales.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/demonstrations"))
    parser.add_argument("--integrantes", type=Path, default=Path("integrantes.txt"))
    parser.add_argument("--layouts-dir", type=Path, default=Path("configs/layouts"))
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    args = parser.parse_args()

    if args.json:
        recordings = collect_recordings(args.data_dir)
        complete = [r for r in recordings if r.complete]
        scenario_counts = Counter(r.layout_name for r in complete)
        summary = {
            "complete_recordings": len(complete),
            "target_recordings": TARGET_RECORDINGS,
            "complete_scenarios": sum(1 for count in scenario_counts.values() if count >= RECORDINGS_PER_SCENARIO),
            "target_scenarios": TARGET_SCENARIOS,
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "integrantes_exists": args.integrantes.exists(),
        }
        print(json.dumps(summary, indent=2))
        return

    print(
        build_progress_report(
            data_dir=args.data_dir,
            integrantes_path=args.integrantes,
            layouts_dir=args.layouts_dir,
        )
    )


if __name__ == "__main__":
    main()
