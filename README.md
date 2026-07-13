# Overcooked-AI — Agente de competencia (Deep Learning)

Agente **híbrido** para la competencia Overcooked-AI del curso: un **planner robusto**
(sin aprendizaje, generaliza a cualquier layout) como piso garantizado, y **especialistas
PPO por layout** como techo, unidos por un selector con fusibles de seguridad. La entrega
es `policies/student_agent.py`, compatible con el loader oficial (`src/policy_loader.py`,
tipo `python_class`).

## Arquitectura

- **Planner** (`policies/planner_agent.py`): FSM + BFS precomputado, < 0.1 ms/acción.
  Incluye **modo receta** (layouts multi-ingrediente como counter_circuit, que usa
  tomates), anti-deadlock/oscilación y detección de **cocinas disjuntas**.
- **Especialistas PPO** (`models/<layout>/best.zip`, SB3 + CNN `SmallGridCNN`): receta
  M3 = BC warm-start del planner + PPO 8M vs población `solo_heavy`. Solo se despliegan
  si `models/<layout>/enabled` existe (lo escribe `scripts/enable_model.py` únicamente
  si el modelo NUNCA empeora al planner — fusible anti-regresión).
- **Selector** (`policies/student_agent.py`): sonda de cooperación (planner hasta que el
  compañero demuestre cooperar → PPO), fallback por **terrain-hash** (identifica el layout
  por el hash del terreno aunque el arnés no lo nombre), watchdog anti-congelamiento y
  fusible de latencia (60 ms) → siempre degrada al planner.

## Resultados (escenarios E1-E3, revelados 2026-07-13)

Sopas promedio del planner contra el compañero real de cada escenario (gate_seeds × swap):

| Escenario | Layout | Compañero | Sopas | Umbral |
|---|---|---|---|---|
| E1 | asymmetric_advantages | greedy | **6.5** | ≥1 ✅ |
| E2 | coordination_ring | greedy+sticky | **3.8** (8.0 vs greedy limpio) | ≥2 prom ✅ |
| E3 | counter_circuit 🍅 | greedy+sticky+random | **9.7** (13.5 vs greedy) | ≥2 prom ✅ |

Gate de entrega **G8: PASS 18/18** (6 layouts × 3 compañeros, con y sin swap; 0 timeouts,
0 acciones inválidas). 11 modelos PPO entrenados el día de competencia no superaron este
piso → E1-E3 se juegan con el planner. Detalle completo: `docs/TRAINING_PROGRESS.md`.

## Cómo correr

```bash
source scripts/env.sh                      # módulos + venv (clúster)
# evaluación
python -m scripts.planner_baseline --layout counter_circuit
python -m scripts.check_vs_sticky --layout coordination_ring --agent planner
python -m evaluation.run_gate --gate G8    # gate integral de la entrega
# entrenamiento (día de competencia: layout nuevo → PPO habilitado, todo automático)
bash scripts/prepare_new_layout.sh <builtin|archivo.layout> [seeds] [steps]
# entrenamientos empaquetados (N por job SLURM; ver docs/CLUSTER_NOTES.md §3.5)
sbatch --export=ALL,JOBS=training/jobs_packed_tesis.txt sbatch/train/run_train_packed.sh
```

## Documentación

- `PLAN.md` — plan maestro (fases, gates G0-G8, reglas anti-trampa, árboles de decisión).
- `docs/TRAINING_PROGRESS.md` — estado consolidado de modelos y resultados (tablas).
- `docs/CLUSTER_NOTES.md` — recetas SLURM del clúster khipu (QOS, bug del epilog, packed).
- `goals/PROGRESS.md` — bitácora por gate/día.

---

# Recolector de demostraciones humanas (subsistema original)

Menú interactivo para jugar partidas, grabar demostraciones y preparar datasets de
Imitation Learning.

## Uso

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt   # RL adicional: requirements-rl.txt
python -m src.game_menu                     # menú interactivo
# alternativas directas:
python -m src.run_game --config configs/play.yaml
python -m src.collect_demonstrations --config configs/collect_demonstrations.yaml
python -m src.dataset_progress              # ver progreso de grabaciones
```

**Controles:** mover = flechas o `W/A/S/D` · interactuar/tomar/dejar/servir = `Space`,
`E` o `Enter` · cancelar = `Escape` o `Q`.

**Configuración** (`configs/collect_demonstrations.yaml`): `environment.horizon` (duración),
`environment.layout_name` (oficial) o `layout_file` (custom en `configs/layouts/`),
`policies.agent_0.name` (agente automático: `stay`, `random_motion`, `greedy_full_task`),
`policies.agent_1.name: human_keyboard`, `data_collection.output_dir`.

**Entrega de datos:** `data/`, `outputs/` y `resultados/` están en `.gitignore`. Para
entregar, carpeta aparte con los `.pkl` de `data/demonstrations/`, `integrantes.txt` y
los `.layout` custom usados.

**Notas:** el warning de Gym al iniciar no bloquea nada. Si las teclas no responden,
mantén presionada la tecla de movimiento (el control captura también pulsaciones rápidas).
