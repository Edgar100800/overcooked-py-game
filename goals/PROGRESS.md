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

## Entrenamiento real A100 (job 46045) — 3 modelos, 1 fallo de infra

Array de 3 tasks x 5M steps en ag001 (shards MPS). Task 0 COMPLETED (5M); tasks 1,2
FAILED por **SIGKILL (0:9)** a 4.8M/4.6M, 1 s despues de que la task 0 termino. Los 3
`best.zip` quedaron intactos (40537 / 40550 / 33527).

**Causa raiz:** NO fue OOM (RSS 3.2GB de 24GB) ni error de codigo (sin traceback). Las 3
tasks compartian UN MIG 1g.5gb via shards MPS; al terminar la task 0, el epilog de
limpieza del nodo (`/etc/slurm/slurm.epilog.clean`) tiro el servidor MPS compartido y
mato a las otras. Es la fragilidad conocida de varios jobs concurrentes sobre una GPU
compartida por MPS. (Detalle completo en el ADDENDUM del plan.)

**Fix aplicado:** entrenamiento en **CPU** por defecto (Overcooked es CPU-bound, el CNN
es diminuto -> misma velocidad; elimina toda la clase de fallos MPS/epilog). sbatch ->
`--partition=standard --device cpu` (validado: job de prueba COMPLETED en n003).
Alternativa A100 documentada: rebanadas MIG DEDICADAS + `--array=..%2`. Ademas
`callbacks.py` ahora guarda `last.zip` en cada eval (resiliencia ante SIGKILL).

## G6 — PASS (2026-07-11, modelos reales 5M)
- cramped_room (seed1, 40550): 6.0 sopas vs greedy, p99 1.06ms.
- asymmetric_advantages (seed0, 33527): 3.0 sopas vs greedy, p99 1.19ms.
Los modelos PPO reales SI aprenden a entregar sopas (mucho mejor que el smoke).

## G7 — decision "planner" (2026-07-11, modelos reales 5M)
El planner robusto SIGUE ganando, sobre todo en el rol invertido (swap):
- cramped_room: sin swap PPO 60271 ~= planner 60221 (gana por poco); CON swap PPO 60310
  < planner 70374. No supera en ambos roles -> no habilita.
- asymmetric: sin swap PPO 60471 < planner 80431; CON swap PPO 0 (no aprendio player-1)
  < planner 50426. No habilita.
Decision valida (PLAN §11 G7): selector queda en planner. Ningun `enabled` escrito.
Mejora futura: mas steps / tecnicas de rol (el PPO flojea como player-1 en swap).

## G8 — PASS FINAL (2026-07-11)
Selector (modo planner, ningun PPO habilitado) en 6 layouts x {greedy, greedy_eps,
random_motion} x swap = 18 celdas, 0 FAIL, 0 timeouts, 0 invalidas. **ENTREGA VALIDADA.**

**RESUMEN FINAL:** G0-G6 PASS, G7 = "planner" (valido), G8 PASS. La entrega es el planner
robusto (4-8 sopas/layout, <0.1ms, 0 timeouts) en TODOS los escenarios incl. 5-6
desconocidos. El PPO quedo entrenado y evaluado pero no habilitado por no superar al
planner (el fusible anti-regresion funciono).

## 🏆 G8 PASS con PPO ACTIVO — primer modelo robusto habilitado (2026-07-11)

Tras 19 entrenamientos (population, greedy, greedy_heavy, solo_heavy, curriculum, BC
warm-start, self-play FCP-lite), el hallazgo: NINGUNA receta de entrenamiento retiene la
habilidad de soloear en evaluacion determinista (todas dan 0 vs random_motion; el clon BC
con acc 0.992 la pierde por covariate shift y el PPO la olvida).

**Solucion ganadora: sonda de cooperacion en el StudentAgent** (ad-hoc teaming, PLAN §15):
planner hasta que el companero sostenga un objeto; PPO desde entonces. random/stay jamas
sostienen -> planner completo (= score del planner); greedy lo hace en ~5 pasos -> PPO.

- Habilitado: custom_zigzag M3 (BC warm-start del planner + PPO solo_heavy 8M, seed300).
- enable-check: greedy 70286 vs planner 36287 (+94%) | eps 47410 vs 41879 | random = planner.
- **G8 FINAL: PASS 18/18 celdas con PPO activo** (+34000 pts sobre el planner en la celda
  zigzag-vs-greedy). El selector nunca empeora al planner en ninguna celda.
- Backlog en curso (BC custom_room, selfplay zigzag, BC dual_pots) puede habilitar mas
  layouts con el mismo mecanismo.

## 🏁 Preparacion dia-de-competencia — ENSAYO GENERAL EXITOSO (2026-07-12)

Formato revelado por el profe: E1-E3 layouts el domingo (greedy/sticky/eps), E4 lunes
(random_motion), E5-E6 en vivo (agente de otro grupo). Torneo por puestos. Preparacion:

1. **Fallback por terrain-hash en StudentAgent**: si el arnes no pasa layout en el config,
   se detecta por hash del terreno (`models/<key>/terrain.key`) y el PPO se activa igual
   (cache de modelos por proceso + precarga en __init__, fuera del SIGALRM). Verificado:
   config {} = mismo score que config completo, 0 timeouts. G8 sigue PASS 18/18.
2. **Fix del planner (hallado por el ensayo)**: detector de oscilacion (osc_window=12) —
   dos agentes deterministas "bailando" en espejo en un pasillo de 1 celda quedaban 0
   sopas para siempre; ahora detour. Rehearsal 0.5→1.0 sopas vs greedy; costo neto en los
   6 layouts: -1002 de 956k (2 celdas cambian >100, una MEJORA).
3. **Playbook `scripts/prepare_new_layout.sh`**: .layout nuevo → baseline (aborta si
   planner no llega a 1 sopa) → BC data → 2 seeds M3 en SLURM (elige nodo/cuenta libre,
   1 job/nodo) → enable-check robusto → habilita → report.md. TODO automatico.
4. **Ensayo cronometrado** en `rehearsal_kitchen` (layout inedito, corredor unico):
   **291 min end-to-end sin intervencion** → seed400 ROBUSTO y HABILITADO:
   greedy 50375 vs planner 10583 (**+376%**) | eps 35534 vs 17922 (+98%) | random =.
   Con config {} (hash): 50375, identico. **El pipeline del domingo esta validado.**
- Backlog evaluado: custom_room s301 ❌, dual_pots s300 ❌ (gana greedy, pierde eps),
  zigzag s301 selfplay ✅ robusto pero < campeon seed300 → respaldo. Campeon intacto.

Plan del domingo: `bash scripts/prepare_new_layout.sh <escenarioX>.layout` por cada
layout revelado (~5h por layout, corren en paralelo en nodos distintos). Lunes E4:
`python -m scripts.planner_baseline --layout-file <e4>.layout` (2 min, el planner basta).

<!-- Las entradas de gate se agregan debajo a medida que run_gate.py los ejecuta -->
