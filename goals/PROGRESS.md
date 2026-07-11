# PROGRESS.md — Bitácora humana

Entrada por gate (pase o falle): qué se hizo, resultado numérico, decisión, commit.
Las reglas anti-trampa (PLAN §10) son no negociables.

---

## Setup inicial (scaffold)

- **Entorno:** `.venv` con `python3/3.10.2` + `cuda/11.8`. numpy 1.26.4, sb3 2.9.0,
  gymnasium 1.3.0, torch 2.7.1+cu118 (A100-compat). Construir con `scripts/setup_venv.sh`.
  Para uso interactivo: `source scripts/env.sh` (carga módulos + activa venv; el `.venv`
  necesita el módulo cargado por su libpython).
- **Infra A100:** nodo `ag001` con MIG ya expuesto por SLURM (no requiere sudo).
  Cuenta `tesis`/`a-tesis`. `DELIVERY_REWARD` = 20 (verificado en runtime).
- **Código:** planner (`policies/planner_agent.py`), arnés `evaluation/`, PPO `training/`,
  selector `policies/student_agent.py`, sbatch `sbatch/`.

## G0 — PASS (2026-07-11)
greedy(P0) vs stay en cramped_room, gate_seeds. mean_soups=1.0, 0 timeouts,
p99=0.07ms. La detección de entregas (sparse>0) funciona.

## G2 — PASS (2026-07-11)
Planner vs stay en cramped_room, con y sin swap. mean_soups=2.5 (≥1), 0 timeouts,
p99=0.09ms. Nota: la celda swap da 0 sopas porque el compañero-stay ocupa la única
celda adyacente al dispensador de platos (deadlock inevitable con bloqueador fijo);
promedia 2.5 igual. Con compañero móvil (G3/G4) no ocurre.

## G3 — PASS (2026-07-11)
Planner vs greedy_full_task en 6 layouts, con y sin swap. mean_soups por layout:
cramped 4.1, asymmetric 6.5, coordination_ring 8.0, custom_room 5.4,
custom_dual_pots 5.5, custom_zigzag 4.7. Todos ≥2, 0 timeouts. verify_ok.

## G4 — PASS (2026-07-11)
Planner vs random_motion (autosuficiente) en 6 layouts, con y sin swap. mean_soups:
cramped 4.8, asymmetric 5.0, coordination_ring 3.1, custom_room 4.7,
custom_dual_pots 6.0, custom_zigzag 5.0. Todos ≥1, 0 timeouts. verify_ok.

**Estado:** G0,G2,G3,G4 aprobados (planner). Falta G1 (freeze), G5-G7 (PPO), G8
(selector). G1 se corre al final, tras estabilizar run_gate con training/ y student.

<!-- Las entradas de gate se agregan debajo a medida que run_gate.py los ejecuta -->
