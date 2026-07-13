# PLAN.md — Agente Overcooked-AI (Proyecto Final Deep Learning)

> Archivo de planificación para Claude Code, diseñado para **ejecución autónoma nocturna** vía `/goal`.
> Repo base: `Edgar100800/overcooked-py-game`. Leer completo antes de escribir código.
> **Regla de oro: ningún gate se considera aprobado sin un rollout real verificado (ver §10). Está PROHIBIDO modificar `evaluation/` o los seeds de validación.**

---

## 1. Contexto de la competencia

- Entorno: Overcooked-AI (`overcooked_ai_py`), 2 agentes cooperando para entregar sopas.
- Score oficial: `10000·sopas + 10·(horizon − t_última_sopa) + (horizon − t_primera_sopa) − penalización`.
  Las sopas dominan; el tiempo desempata. Penalización = `min(100·timeouts, 5000)`. Si sopas = 0 → score 0.
- Límite por acción: **100 ms** (`max_action_time_ms: 100`). La inferencia DEBE ser < 50 ms (objetivo < 20 ms).
- 3 seeds por escenario, se promedia. Algunos escenarios evalúan **cambio de rol** (agente puede ser index 0 o 1).

| Escenario | Layout | Compañero | Estado |
|---|---|---|---|
| 1 | ✅ `asymmetric_advantages` | `greedy_full_task` | preparado 07-13 → planner (6.5 sopas) |
| 2 | ✅ `coordination_ring` | `greedy_full_task` + sticky actions | preparado 07-13 → planner (8.0 / 3.8 vs sticky) |
| 3 | ✅ `counter_circuit` 🍅 TOMATES | `greedy_full_task` + sticky + random | preparado 07-13 → planner modo receta (**13.5 sopas**) |
| 4 | revelado lunes temprano | `random_motion` (el agente hace TODO solo) | baseline planner el lunes (2 min) |
| 5 | revelado EN competencia | agente de otro grupo | planner + fallback terrain-hash |
| 6 | revelado EN competencia | agente de otro grupo | planner + fallback terrain-hash |

> E1-E3 revelados 2026-07-13 con rúbrica por puestos (el ranking define la nota; E3
> clasifica solo top-12, E4 top-8). Detalle y baselines en `docs/TRAINING_PROGRESS.md` §0-§2.
> counter_circuit: ninguna orden es la sopa de 3 cebollas → exigió el modo receta (§4.8).

**Estrategia: agente híbrido.**
1. **Planner robusto** (sin aprendizaje, generaliza a cualquier layout) = piso garantizado en TODOS los escenarios.
2. **PPO por layout** cuando el layout es conocido = techo competitivo en escenarios 1-4.
3. `StudentAgent` selector: usa modelo PPO si existe y está validado para el layout; si no, planner. Fusible de latencia y de excepciones → degradar a planner.

---

## 2. Lo que ya existe en el repo (reutilizar, no reescribir)

- `src/runner.py` — loop con role swap (`swap_agent_positions`), seeds por episodio, logging.
- `src/policy_loader.py` — carga `python_class` con clase `StudentAgent`. **La entrega final debe ser compatible con este loader.**
- `src/policy_wrappers.py` — `StudentAgentAdapter` (act(obs)->int), `EpsilonActionWrapper` (random actions, escenario 3), `SafeActionWrapper` (timeouts, expone `timeout_count`).
- `src/observations.py` — `ObservationBuilder`: `state` (estado crudo + mdp), `featurized` (vector), `lossless_grid` (tensor HxWxC).
- `policies/basic_policies.py` — `GreedyFullTaskPolicy` = **el compañero oficial de escenarios 1-3**. Entrenar contra él.
- `src/environment.py` — layouts oficiales y custom `.layout`.
- Acciones: `0=norte, 1=sur, 2=este, 3=oeste, 4=stay, 5=interact` (`src/constants.py`).

---

## 3. Estructura de carpetas objetivo

```
overcooked-py-game/
├── policies/
│   ├── student_agent.py        # ENTREGA FINAL (selector híbrido)
│   ├── planner_agent.py        # Fase 1
│   └── basic_policies.py       # existente
├── training/
│   ├── gym_env.py              # wrapper Gymnasium single-agent
│   ├── partner_population.py   # pool de compañeros
│   ├── sticky_wrapper.py       # StickyActionWrapper (repite acción previa con prob p)
│   ├── reward_shaping.py
│   ├── train_ppo.py
│   └── callbacks.py            # eval periódica con score oficial
├── models/<layout>/best.zip
├── evaluation/                 # ★ CONGELADO tras crearse (ver §10)
│   ├── official_score.py
│   ├── run_gate.py             # ejecuta un gate y emite artefactos
│   ├── verify.py               # re-computa el score desde step logs crudos
│   ├── gate_seeds.json         # seeds held-out (checksum en freeze.sha256)
│   └── freeze.sha256
├── goals/
│   ├── GOALS.md                # definición de gates (este §11)
│   ├── progress.json           # estado por gate (escrito SOLO por run_gate.py)
│   └── PROGRESS.md             # bitácora humana (qué se probó, qué funcionó, qué no)
├── configs/  y  src/           # existentes
```

---

## 4. FASE 1 — Planner robusto (`policies/planner_agent.py`) — PRIORIDAD MÁXIMA

1. Observación tipo `state` (estado crudo + mdp). Configurar `observation.type: state`.
2. Pathfinding BFS/A* precomputado en `reset()` desde `mdp.terrain_mtx`: distancias celda→estación (pots, dispensers, serving). Cachear todo.
3. Máquina de estados de sub-tareas: sopa→servir; plato→pot listo/casi listo; onion→pot no lleno; vacío+sopa lista→plato; vacío→onion.
4. Modelado del compañero: inferir su sub-tarea (posición + objeto) y tomar la complementaria. Con `random_motion`/`stay`: completar el ciclo COMPLETO solo, compañero = obstáculo móvil.
5. **Anti-deadlock:** N pasos sin moverse → ruta alternativa → si no hay, retroceso lateral aleatorio 1-2 pasos → reintentar. Causa #1 de 0 sopas en layouts angostos.
6. Independiente del índice (`agent_index` viene en el obs dict). Probar SIEMPRE con `swap_agent_positions: true`.
7. Presupuesto: todo precomputado en reset; `act()` solo lookups. Objetivo < 5 ms.
8. **Modo receta (2026-07-13):** si NINGUNA orden del layout es la mono-sopa clásica
   (p.ej. counter_circuit: todas llevan tomate), apuntar a la orden válida más rápida,
   rellenar cada olla con lo que le FALTA, iniciar cocción al completar la receta, y
   cocinar-y-entregar a valor 0 las ollas envenenadas por compañeros solo-cebolla para
   liberarlas. Se desactiva solo en layouts clásicos (regresión cero verificada).
9. **Cocinas disjuntas (2026-07-13):** flood-fill desde los starts + chequeo de
   autosuficiencia de AMBAS regiones (forced_coordination queda excluido); si el layout
   es tipo asymmetric_advantages, ignorar el plato del compañero cuando hay 2+ ollas
   listas (cada uno tiene su propio dispensador; evita diferir la 2ª sopa).

---

## 5. FASE 2 — Pipeline RL (PPO)

- Dependencias: `gymnasium`, `stable-baselines3>=2.0`, `torch` (CPU). Referencia de arquitectura: PantheonRL (SB3 + Overcooked + diversidad de compañeros), human_aware_rl (deprecado, solo referencia de hiperparámetros/shaping), HAHA (ad hoc teaming).
- `training/gym_env.py`: `OvercookedEnv` como entorno single-agent; el compañero (política fija/muestreada) se ejecuta dentro de `step()`. `action_space=Discrete(6)`. **Randomizar índice del agente (0/1) en cada `reset()`** → aprende ambos roles.
- Observación: empezar `featurized` + MlpPolicy (rápido en M4); escalar a `lossless_grid` + CNN solo si se estanca (§12-D).
- `partner_population.py`, sampleo por `reset()`: greedy limpio 35%, greedy+sticky(p≈0.25)+ε(0.1-0.2) 25%, random_motion 20%, self-play (checkpoints congelados propios) 15%, stay 5%. Recetas por CLI (`--partner`): population / greedy / greedy_heavy / solo_heavy (M3) / **sticky_heavy** (2026-07-13: sticky PURO 35% — kind `greedy_sticky` — para el compañero exacto de E2).
- `reward_shaping.py`: `r = sparse + coef·shaped_r_del_agente` (shaped nativos del MDP: pot=3, dish=3, soup_pickup=5), `coef` con annealing 1.0→0.0 en el primer 60% de timesteps.
- `train_ppo.py`: PPO SB3, `n_envs=8-16` (SubprocVecEnv), `n_steps=400`, `batch=2000`, `lr=3e-4` decay, `ent_coef=0.02→0.001`, `gamma=0.99`, `gae_lambda=0.98`, 5-10M timesteps/layout (la literatura reporta ~8M para converger en cramped_room). CLI: `--layout --timesteps --out`.
- Callback: cada N steps, 5 episodios con **score oficial** vs greedy_full_task; guardar `best.zip` por score oficial, NO por reward de entrenamiento.
- Inferencia: cargar modelo una vez en `__init__`, `predict(deterministic=True)`, `torch.set_num_threads(1)`, CPU. Medir latencia.

---

## 6. FASE 3 — Evaluación y agente final

- `evaluation/official_score.py`: fórmula EXACTA. `t_primera/t_última` extraídos de los timesteps donde el sparse reward del env fue > 0 (entrega de sopa); timeouts desde `SafeActionWrapper.timeout_count`.
- `evaluation/run_tournament.py` (o dentro de run_gate): matriz {planner, ppo} × {greedy, greedy+sticky, greedy+sticky+ε, random_motion, planner_propio} × {con/sin swap} × seeds.
- `policies/student_agent.py`: selector con detección de layout (config o hash de `terrain_mtx`), try/except + fusible de latencia (>60 ms → planner por el resto del episodio). Nunca devolver acción inválida. Verificar con `python -m src.run_game --config configs/evaluate.yaml`.

---

## 7. Opcional (solo si sobra tiempo)
Warm-start por behavior cloning sobre trayectorias generadas por el **planner** (miles gratis, no humanas), luego PPO encima. Se activa solo desde §12-D.

---

# PARTE II — SISTEMA DE VALIDACIÓN AUTÓNOMA (para corrida nocturna con /goal)

## 8. Filosofía

El agente (Claude Code) trabajará horas sin supervisión. El riesgo es doble: (a) que avance sin validar y a la mañana nada funcione, (b) que "valide" con tests triviales/mockeados que se auto-aprueban. Por eso:

- El progreso se mide EXCLUSIVAMENTE con **gates** (G0-G8), cada uno una prueba de concepto ejecutable con criterio numérico.
- Un gate solo lo aprueba `evaluation/run_gate.py` ejecutando **rollouts reales** en el entorno.
- `progress.json` lo escribe SOLO `run_gate.py`. Editarlo a mano = trampa = corrida inválida.
- Después de cada gate (pase o falle), Claude Code escribe una entrada en `goals/PROGRESS.md`: qué hizo, resultado numérico, decisión tomada, y hace `git commit`.

## 9. Loop de trabajo nocturno (algoritmo para Claude Code)

```
1. Leer goals/progress.json → identificar el primer gate no aprobado.
2. Implementar/arreglar lo mínimo necesario para ese gate.
3. Ejecutar: python -m evaluation.run_gate --gate GX
4. Si PASA: registrar en PROGRESS.md, git commit -m "GX passed: <métrica>", ir al siguiente gate.
5. Si FALLA: consultar el árbol de decisión (§12) del gate, aplicar la SIGUIENTE alternativa
   no probada (registrar cuál en PROGRESS.md), volver a 2.
6. Si se agotan las alternativas de un gate: marcarlo BLOQUEADO en PROGRESS.md con diagnóstico
   (logs, hipótesis), y continuar con gates independientes (p.ej. G7 no depende de G4-G6).
   NUNCA borrar el trabajo de un gate bloqueado.
7. Cada 90 min como máximo debe existir un commit. Si un entrenamiento largo corre,
   aprovechar para preparar el siguiente gate en paralelo (código, no validación).
```

Regla de presupuesto: si un mismo error se repite 3 veces con el mismo enfoque, cambiar de enfoque (árbol §12), no insistir.

## 10. Reglas anti-trampa (NO NEGOCIABLES)

1. **`evaluation/` se congela** en cuanto G1 pasa: se genera `freeze.sha256` con los checksums de `official_score.py`, `run_gate.py`, `verify.py`, `gate_seeds.json`. `run_gate.py` verifica los checksums al inicio y aborta si no cuadran. Cambios a `evaluation/` después del freeze solo con aprobación humana explícita (dejarlo pendiente en PROGRESS.md).
2. **Seeds held-out:** `gate_seeds.json` contiene 2 conjuntos: `dev_seeds` (visibles, para iterar) y `gate_seeds` (para aprobar gates). Los gates se corren SIEMPRE con `gate_seeds`. Prohibido entrenar/ajustar usando `gate_seeds`.
3. **Verdad = entorno:** sopas y tiempos se computan desde los eventos del env (sparse reward > 0 en el step log), nunca desde contadores del agente, prints, o valores retornados por la política. `verify.py` recalcula el score desde el step log crudo y debe coincidir con lo reportado (tolerancia 0).
4. **Prohibido:** mockear `OvercookedEnv` en validaciones; reducir `horizon` (<250) en gates; tests unitarios que sustituyan rollouts como criterio de gate; capturar excepciones para reportar éxito; escribir `progress.json` desde otro script; comentar/eliminar asserts de `run_gate.py`.
5. **Artefactos obligatorios por gate:** `outputs/gates/GX_<timestamp>/` con `results.json` (score por episodio, sopas, t_primera, t_última, timeouts, latencia p50/p99), step logs crudos, y el config exacto usado. Sin artefactos → gate no aprobado, aunque el número se haya visto en consola.
6. Los tests unitarios (pytest) SÍ son bienvenidos para desarrollo (pathfinding, shaping, wrapper gym con `check_env` de SB3), pero **jamás cuentan como aprobación de gate**.

## 11. GATES — metas medibles (copiar a goals/GOALS.md)

Config de gate salvo indicación: horizon 250, `max_action_time_ms: 100`, 5 episodios con `gate_seeds`, promediando; "swap" = correr adicionalmente con `swap_agent_positions: true`.

| Gate | Prueba de concepto | Criterio de aprobación (medido por run_gate.py) |
|---|---|---|
| **G0** | Smoke del entorno | `run_game` corre con template `stay` vs `greedy_full_task` en `cramped_room` sin excepciones; el step log registra ≥1 sopa del greedy (prueba que la detección de entregas funciona) |
| **G1** | Score oficial correcto | `official_score.py` reproduce a mano 3 casos sintéticos (0 sopas→0; caso con timeouts; caso normal) Y `verify.py` == reporte en un rollout real. Al pasar → **freeze de evaluation/** |
| **G2** | Planner mueve e interactúa | Planner solo (compañero `stay`) en `cramped_room`: ≥1 sopa promedio, 0 timeouts, latencia p99 < 50 ms |
| **G3** | Planner competitivo | Planner + `greedy_full_task` en 6 layouts (cramped_room, asymmetric_advantages, coordination_ring + 3 custom del repo): ≥2 sopas promedio por layout, 0 timeouts, **con y sin swap** |
| **G4** | Planner autosuficiente (esc. 4) | Planner + `random_motion` en los 6 layouts: ≥1 sopa promedio por layout, con y sin swap |
| **G5** | Gym env válido | `check_env` de SB3 pasa; 3 episodios aleatorios sin excepciones; PPO 50k steps corre sin NaN y el reward shaped promedio SUBE vs política aleatoria |
| **G6** | PPO aprende (smoke) | PPO en `cramped_room` vs greedy, ≥2M steps: score oficial > 0 (≥1 sopa) en gate_seeds |
| **G7** | PPO supera al planner | En ≥1 layout: score oficial PPO > score planner (mismos seeds, mismo compañero), latencia p99 < 50 ms, con y sin swap. Si no, el selector queda en planner para ese layout (decisión válida, documentar) |
| **G8** | Entrega integrada | `student_agent.py` vía `configs/evaluate.yaml` en los 6 layouts × {greedy, greedy+sticky+ε, random_motion} × swap: 0 crashes, 0 acciones inválidas, 0 timeouts, y score ≥ al del planner puro en cada celda (el selector nunca empeora) |

**Definición de "avanzar":** el número de gates aprobados en `progress.json` (escrito por run_gate) es la única métrica de avance de la noche. Meta mínima de una noche: G0-G4 + G8 en modo solo-planner (= entrega funcional garantizada). Meta ideal: G0-G8.

## 12. Árboles de decisión (alternativas si algo falla)

**A. G2/G3/G4 — el planner entrega 0 sopas o pocas:**
1. Revisar step log: ¿llega al pot pero no interactúa? → bug de orientación (interact requiere estar ADYACENTE y MIRANDO a la estación; la acción previa debe orientar).
2. ¿Se queda quieto muchos pasos? → deadlock con el compañero → afinar detector (umbral N=3-5) y desvío lateral.
3. ¿Va a estaciones equivocadas? → verificar mapeo de `terrain_mtx` (P, O, D, S, X) y que las distancias se calculen a la celda transitable ADYACENTE a la estación, no a la estación misma.
4. ¿Falla solo con swap? → hay un index 0 hardcodeado; buscar y parametrizar.
5. Alternativa mayor: reemplazar la FSM propia por `GreedyHumanModel` de overcooked_ai_py (usa `env.mlam`, planners oficiales) envuelto con el anti-deadlock propio; comparar ambos y quedarse con el mejor score.
6. ¿Latencia > 50 ms? → el MLAM oficial puede ser lento en su primer cómputo: precomputarlo en `__init__`/`reset` (fuera del primer `act`), o volver a la FSM ligera.

**B. G5 — gym env inválido / PPO explota:**
1. `check_env` falla → revisar dtype float32, shapes constantes, `reset()` retorna (obs, info) (API Gymnasium).
2. NaN en pérdidas → clip de reward, `lr=1e-4`, verificar que la obs no contenga inf.
3. Episodios no terminan → `terminated` en horizon, `truncated` correcto.

**C. G6 — PPO no aprende (score 0 tras 2M steps):**
Aplicar EN ORDEN, un cambio a la vez, 1M steps de prueba por cambio:
1. Subir shaping: coef fijo 1.0 sin annealing (reactivar annealing después).
2. Subir exploración: `ent_coef=0.05` inicial.
3. Compañero fijo greedy 100% (sin población) hasta lograr la primera sopa; reintroducir población después.
4. Cambiar obs: featurized → `lossless_grid` + CNN pequeña (3 conv de 32-64 filtros).
5. Warm-start BC con 200-500 episodios del planner (§7), luego PPO.
6. Recompensa de acercamiento (potential-based sobre distancia a la sub-tarea del shaped event más cercano) — con cuidado, retirarla con annealing.
7. Si nada funciona en el presupuesto de la noche: G7 se resuelve "planner gana" y se documenta. NO es fracaso: la entrega sigue garantizada.

**D. G7 — PPO aprende pero no supera al planner:**
1. Más timesteps (5M→10M) solo en el/los layouts objetivo.
2. Annealing del shaping más largo (80%).
3. Población con más peso de self-play (robustez) o del compañero exacto del escenario objetivo (especialización) — elegir según el escenario que se quiera atacar.
4. lossless_grid + CNN si aún está en featurized.
5. Fine-tune desde el mejor checkpoint contra el compañero exacto del escenario (transfer corto de 1-2M steps).

**E. G8 — el selector empeora o crashea:**
1. Cualquier excepción/latencia en PPO → el fusible debe degradar a planner (verificar que el fusible realmente se dispara con un test de inyección de fallo — este test unitario sí vale, pero G8 se aprueba con rollouts).
2. Score selector < planner en alguna celda → el criterio de selección por layout está mal calibrado: exigir que PPO supere al planner en gate_seeds ANTES de habilitarlo para ese layout.

**F. Infraestructura (cualquier gate):**
- Entrenamiento demasiado lento en M4 → bajar `n_envs`, usar featurized, horizon de ENTRENAMIENTO 200 (los gates siguen en 250); si sigue siendo inviable, considerar Colab/GPU para entrenar y traer el `.zip` (la inferencia queda en CPU).
- Import errors de overcooked_ai → respetar `numpy<2` del requirements; no actualizar dependencias del entorno por resolver un bug propio.

## 13. Cronograma vs revelación de layouts

| Paso | Tarea | Done |
|---|---|---|
| 1 | G0-G4 (planner) | gates aprobados |
| 2 | G8 modo solo-planner | entrega funcional garantizada para CUALQUIER escenario |
| 3 | G5-G6 (pipeline PPO, smoke en cramped_room) | gates aprobados |
| 4 | G7 en los 3 layouts custom (ensayo general) | tabla comparativa |
| 5 | **Domingo:** layouts 1-3 → 3 entrenamientos en paralelo + G7 por layout | ✅ 2026-07-13: 3 playbooks + 2 packed = 11 modelos M3; NINGUNO supera al planner → decisión "planner" en E1-E3 (G8 PASS 18/18) |
| 6 | **Lunes:** layout 4 → baseline planner (2 min); entrenamiento corto solo si el PPO promete | siguiente paso |
| 7 | Escenarios 5-6 → planner (o PPO si algún modelo generaliza en prueba rápida) | — |

## 14. Riesgos
- Timeout 100 ms → precomputar todo, torch CPU 1 hilo, fusible, medir p99 en cada gate.
- Sobreajuste al compañero → población (§5) + eval contra compañeros no vistos.
- Layouts 5-6 desconocidos → el planner es el plan A.
- Role swap → randomizar índice en entrenamiento; todos los gates corren con swap.
- Reward hacking del agente autónomo → §10.
- Reproducibilidad → seeds fijos, configs versionados junto a cada `.zip`.

## 15. Referencias externas (consultar si se necesita código de ejemplo)
- `HumanCompatibleAI/overcooked_ai` — entorno oficial; `human_aware_rl/` (deprecado) tiene PPO+RLlib, wrappers y shaping de referencia.
- `Stanford-ILIAD/PantheonRL` — SB3 + Overcooked, self-play, torneos, diversidad de compañeros. La referencia más cercana a este plan.
- `StephAO/HAHA` — Hierarchical RL for Ad Hoc Teaming (AAMAS'23), agentes robustos a compañeros desconocidos (relevante para escenarios 5-6).
- Blog Madrona/Overcooked (bsarkar321) — ~8M steps para converger en cramped_room; cross-play contra políticas diversas mejora robustez.
- `DI-engine` (dizoo/overcooked) — otra implementación PPO de referencia.
- Papers si se quiere profundizar: FCP (Fictitious Co-Play), MEP (Maximum Entropy Population) — técnicas de población para zero-shot coordination.

---

## 16. Paralelización de entrenamientos con NVIDIA A100 + MIG

> **ESTADO (2026-07-13): esta sección quedó como referencia.** En producción se usó
> **CPU + jobs empaquetados** (N `train_ppo` dentro de UN job de 30 cpus,
> `sbatch/train/run_train_packed.sh`) — mismo throughput, sin la fragilidad MPS/epilog
> que mató el job 46045. Ver `docs/CLUSTER_NOTES.md` §3.5.

**Principio:** el cuello de botella de Overcooked es la simulación en CPU (Python, ~2k steps/s), no la red. La A100 NO se usa para acelerar un solo entrenamiento sino para correr **muchos entrenamientos independientes en paralelo** (layout × seed × config), uno por instancia MIG.

### 16.1 Setup de MIG (una sola vez, requiere sudo)
```bash
sudo nvidia-smi -mig 1                                # habilitar MIG
nvidia-smi mig -lgip                                  # ver perfiles del hardware
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C      # A100 40GB: 7× 1g.5gb (id 19)
# Alternativas: -cgi 14,14,14 -C (3× 2g.10gb) | -cgi 9,9 -C (2× 3g.20gb)
nvidia-smi -L                                          # anotar UUIDs MIG-xxxx
```
Elegir el número de rebanadas según cores de CPU disponibles (ver 16.3). Para redes pequeñas (CNN de 3 capas) 1g.5gb sobra.

### 16.2 Cambios en `training/train_ppo.py`
- Flag `--device {cpu,cuda}`. En GPU: `PPO(..., device="cuda")`.
- Con GPU disponible, usar `lossless_grid` + CnnPolicy por defecto (la GPU no aporta con MLP diminuto; SB3 incluso recomienda CPU para MLP).
- En cada worker de SubprocVecEnv: `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` (evitar sobre-suscripción de CPU).
- Salidas separadas por job: `models/<layout>/seed<k>_<config>/` con config YAML versionado.

### 16.3 Launcher (`training/launch_parallel.py` o bash)
- Lee una lista de jobs `(layout, seed, config)` y la lista de UUIDs MIG.
- Lanza cada job con `CUDA_VISIBLE_DEVICES=<MIG-UUID>` (un proceso ve UNA sola instancia; MIG no permite combinar rebanadas ni NCCL entre ellas).
- Restricción de CPU: `n_jobs × n_envs ≤ cores físicos`. Ej.: 32 cores → 4 jobs × 8 envs, o 7 jobs × 4 envs. Ajustar `n_envs` por job, no lanzarse a ciegas.
- `nohup`/`tmux` por job, log a archivo propio, y un watcher que escriba cada 30 min a `goals/PROGRESS.md` el último score de eval de cada job.
- Al terminar (o a hora fija), correr G7 sobre TODOS los checkpoints con `gate_seeds` y seleccionar el mejor por layout. Los gates se validan igual (§10): en el env oficial, CPU, con `verify.py`.

### 16.4 Matriz recomendada (domingo, 7 rebanadas)
| Rebanada | Job |
|---|---|
| 1-2 | layout escenario 1 × seeds {0,1} |
| 3-4 | layout escenario 2 × seeds {0,1} |
| 5-6 | layout escenario 3 × seeds {0,1} |
| 7 | experimento libre (shaping coef alterno, población alterna, o layout más difícil con más steps) |

Lunes (escenario 4): reconfigurar a 2-3 rebanadas grandes y correr 2-3 seeds del layout 4 con población dominada por `random_motion`+`stay`.

### 16.5 Rama opcional: simulación en GPU (Madrona)
Existe un port de Overcooked al motor Madrona con throughput de millones de steps/s (miles de entornos en paralelo en GPU). Solo considerarlo si el tiempo de entrenamiento resulta insuficiente tras 16.1-16.4, porque: (a) dependencia externa pesada, (b) riesgo de divergencia de dinámica vs `overcooked_ai_py` oficial. Si se usa: entrenar en Madrona, pero TODA validación de gates en el env oficial (regla §10.3 sin cambios).

### 16.6 Árbol de decisión GPU
1. GPU-util baja (<20%) en cada rebanada con CNN → normal (CPU-bound); no "arreglar", verificar que los cores no estén saturados (`htop`).
2. Jobs se matan por OOM en 1g.5gb → reducir batch/red, o reconfigurar a rebanadas 2g/3g con menos jobs.
3. Throughput total no mejora al agregar jobs → CPU saturada: bajar `n_envs` por job o número de jobs.
4. `CUDA_VISIBLE_DEVICES` con UUID no funciona en contenedor → Docker necesita `--gpus '"device=MIG-UUID"'`.