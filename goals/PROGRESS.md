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

## G5 — PASS (2026-07-11)
Gym env válido: `check_env` de SB3 pasa; PPO smoke (companero greedy fijo, shaping
1.0, ent 0.05) sin NaN; shaped entrenado 2.625 > random 1.5 -> el pipeline aprende.

## Smoke A100 — OK (2026-07-11)
`sbatch sbatch/train/run_smoke_a100.sh` (job 46044) corrio en **ag001, MIG 1g.5gb**,
`torch 2.7.1+cu118 cuda_available True NVIDIA A100-PCIE-40GB`. 151k steps @ 2281 fps,
produjo models/cramped_room/smoke_a100/best.zip. El camino GPU/MIG/cuda funciona.

## G8 — PASS (2026-07-11)
Selector (modo solo-planner, sin modelos habilitados) en 6 layouts x {greedy,
greedy_eps, random_motion} x swap = 18 celdas, 0 FAIL, 0 timeouts, 0 inválidas,
score >= planner puro en cada celda. Fix clave: planner determinista (seed fija 0)
+ compañeros builtin con semilla derivada del episodio (reproducibilidad §14).

## G6 — PASS (2026-07-11, modelo smoke)
PPO smoke (150k) vs greedy en cramped_room: mean_soups=3.0 (>=1), p99=1.26ms. PPO
aprende a hacer sopas. (Con 5M steps reales en A100 el techo es mayor.)

## G7 — FAIL / decision "planner" (2026-07-11, modelo smoke)
PPO smoke NO supera al planner: sin swap ppo~=planner, CON swap ppo=0 (el modelo de
150k no aprendio el rol player-1). Sistema anti-regresion OK: NO escribe models/
cramped_room/enabled -> el selector queda en el planner robusto. Re-correr tras el
entrenamiento real de 5M steps (puede voltear a PASS y habilitar el modelo).

## G1 — PASS (2026-07-11) -> FREEZE
`official_score` reproduce 3 casos sintéticos y `verify.py`==reporte en rollout real.
Se escribio evaluation/freeze.sha256 (4 archivos). evaluation/ CONGELADO (§10.1):
run_gate verifica los checksums al inicio y aborta si cambian.

**Estado noche:** G0-G6,G8 PASS. G7 pendiente de modelo real (5M steps A100). La
entrega funcional esta GARANTIZADA (selector = planner robusto en cualquier layout).

### Proximos pasos (corrida real A100)
1. `sbatch sbatch/train/run_train_ppo.sh` (array 0-6, jobs.txt: layouts esc.1-3 x seeds).
2. Al terminar: `FASE=ppo bash scripts/night_loop.sh` -> G6/G7 por layout (habilita
   los que superen al planner) + re-evalua G8.
3. `sbatch sbatch/eval/run_gate.sh G8` como verificacion final de la entrega.

<!-- Las entradas de gate se agregan debajo a medida que run_gate.py los ejecuta -->
