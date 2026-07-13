# Avance de entrenamientos PPO — Overcooked

Estado consolidado de todos los entrenamientos y sus resultados.
**Actualizado 2026-07-13 (3ª ed. — día de competencia: layouts E1-E3 revelados y preparados).**

> ## 🗓️ DÍA DE COMPETENCIA — resumen en 4 líneas
> Los layouts E1-E3 se revelaron (todos builtin): **E1 `asymmetric_advantages`**,
> **E2 `coordination_ring`**, **E3 `counter_circuit`** (¡usa TOMATES!). El planner ganó
> **modo receta** (counter_circuit: 0 → **13.5 sopas**) y se entrenaron **11 modelos M3**:
> ninguno superó al planner → **la entrega E1-E3 es el planner** (piso altísimo). G8 PASS 18/18.

> ## 🏆 HITO previo: primer PPO robusto HABILITADO (custom_zigzag, M3 BC)
> **G8 PASS 18/18 con PPO activo.** La clave NO fue más entrenamiento sino la **sonda de
> cooperación** en el StudentAgent (ad-hoc teaming, PLAN §15): planner hasta que el
> compañero sostenga un objeto (greedy lo hace en ~5 pasos → PPO el resto; random/stay
> jamás → planner todo el episodio). Resultado en zigzag:
> greedy **70286 vs planner 36287 (+94%)** · eps 47410 vs 41879 · random 50425 = planner.
> Habilitados hoy: `custom_zigzag_kitchen/seed300` y `rehearsal_kitchen/seed400` (ambos
> con terrain.key para el fallback por hash — cubren E5/E6 si el layout coincidiera).

> **Verdad = env.** Todos los scores son con el score oficial de la competencia
> (`10000·sopas + timing − timeouts`) en `gate_seeds`, con y sin swap. `~Ns` = nº de sopas.
> El enable-check (`scripts/enable_model.py`) compara el PPO vs el planner puro contra los
> **3 compañeros** que prueba G8 (greedy, greedy_eps, random_motion). Se **habilita** un
> modelo solo si NO empeora al planner en NINGÚN compañero.

---

## 0. Escenarios y rúbrica oficial (revelado 2026-07-13)

| Esc | Layout | Compañero | Clasificación / notas |
|---|---|---|---|
| 1 | `asymmetric_advantages` | greedy_full_task | ≥1 sopa pasa · 6 pts base, 9 top-5 |
| 2 | `coordination_ring` | greedy + **sticky** | ≥2 sopas prom pasa · 9 base, 12 top-5 |
| 3 | `counter_circuit` 🍅 | greedy + sticky + **random** | solo **top-12** clasifican · 11 base, 14 top-4 |
| 4 | se revela lunes temprano | random_motion | solo top-8 clasifican · 12 base, 16 top-4 |
| 5 | se revela en competencia | agente de otro grupo | solo top-3 al final · 16-17 pts |
| 6 | se revela en competencia | agente de otro grupo | 18 (p3) / 19 (p2) / 20+sublime (p1) |

El ranking por puestos define la nota → **maximizar score, no solo pasar el umbral**.

🍅 **counter_circuit usa recetas con tomate** (onion+tomato bonus cuece en 22 ticks;
también o+o+t y o+t+t) — ninguna orden es la sopa clásica de 3 cebollas. Fue el hallazgo
crítico del día: el planner solo-cebollas hacía 0 sopas hasta el **modo receta** (§2).

---

## 1. Resumen ejecutivo

- **Entrega para E1-E3: el planner robusto** (con modo receta y detección de cocinas
  disjuntas). Ningún PPO de los 11 entrenados hoy lo superó — el fusible anti-regresión
  no habilitó ninguno, y G8 sigue PASS 18/18.
- **El piso es muy alto**: 6.5 sopas en E1, 8.0 en E2 (3.8 vs sticky, la única celda
  floja), **13.5 en E3** — probablemente top-tier en el escenario clasificatorio.
- **PPO habilitados** (sin cambios hoy): custom_zigzag seed300 y rehearsal seed400,
  activables por terrain-hash si E5/E6 reusaran esos terrenos.
- **El muro cambió**: en los layouts custom el muro era "0 vs random"; en E1-E3 el
  muro fue el propio planner (los PPO sí hacen 5-8 sopas, pero el planner hace más).

---

## 2. La barra a superar — planner puro (baseline)

Layouts de competencia (gate_seeds × swap, sopas promedio):

| Layout | vs greedy | vs greedy_eps | vs random | vs **sticky** | vs **sticky+eps** |
|---|---|---|---|---|---|
| **asymmetric_advantages (E1)** | 65428 (6.5s) | 73448 (7.3s) | 50690 (5.0s) | 65481 (6.5s) | 66475 (6.6s) |
| **coordination_ring (E2)** | 80244 (8.0s) | 48682 (4.8s) | 26690 (2.6s) | **38940 (3.8s)** ⚠️ | 37786 (3.7s) |
| **counter_circuit (E3)** | **136154 (13.5s)** | 123545 (12.3s) | 34392 (3.4s) | 101421 (10.1s) | 97861 (9.7s) |
| counter_circuit_o_1order | 60256 (6.0s) | 35748 (3.5s) | 20924 (2.0s) | — | — |

(sticky = greedy+StickyPartner(0.25) puro, el compañero EXACTO de E2; sticky+eps añade
random 0.15, el de E3. Medidos con `scripts/check_vs_sticky.py` sin tocar `evaluation/`.)

Layouts legacy (sin cambios — el modo receta NO altera los layouts de 3 cebollas,
regresión verificada con números idénticos):

| Layout | vs greedy | vs random | nota |
|---|---|---|---|
| cramped_room | 65298 (6.5s) | 50386 (5.0s) | fuerte en ambos |
| custom_room | 60210 (6.0s) | 50389 (5.0s) | medio |
| custom_dual_pots | 56073 (5.5s) | 60570 (6.0s) | medio |
| **custom_zigzag_kitchen** | **36287 (3.5s)** | 50425 (5.0s) | planner débil → ahí ganó el PPO |

**Novedades del planner (2026-07-13):**
- **Modo receta** (multi-ingrediente): solo se activa si NINGUNA orden es la mono-sopa
  clásica. Apunta a la orden válida más rápida, calcula qué le falta a cada olla, inicia
  la cocción al completar la receta, y las ollas "envenenadas" por el greedy solo-cebollas
  las completa (o+o+t) o las cocina-y-entrega a valor 0 para liberarlas.
- **Cocinas disjuntas** (asymmetric): flood-fill + autosuficiencia de ambas regiones
  (forced_coordination queda excluido); ignora al compañero solo si hay 2+ ollas listas
  y él acapara el plato (números medidos idénticos; seguro ante stickies patológicos).

---

## 3. Enfoques probados (compañeros de entrenamiento)

| Enfoque | Mezcla de compañeros | Idea |
|---|---|---|
| **population** (default) | greedy 35 / sticky+eps 25 / random 20 / self-play 15 / stay 5 | balanceado |
| **greedy (100%)** | greedy 100 | especializar (sobreajusta) |
| **greedy_heavy** | greedy 55 / sticky+eps 15 / random 25 / stay 5 | fuerte vs greedy, algo robusto |
| **solo_heavy** (M3) | stay 25 / random 35 / greedy 30 / sticky+eps 10 | **60% no-coop → fuerza solo** |
| **curriculum** | fase-1: stay/random 50/50 → fase-2: solo_heavy | aprende solo primero |
| **sticky_heavy** 🆕 | **sticky puro 35** / sticky+eps 15 / greedy 25 / random 20 / stay 5 | especialista E2 (no ganó ni su celda) |

🆕 2026-07-13: kind `greedy_sticky` PURO (sin epsilon) en la población + receta
`--partner sticky_heavy` en train_ppo.

---

## 4. Tabla maestra de modelos

`eval/enable` = score del student (con sonda) vs greedy en el enable-check (o proxy de
entrenamiento para los históricos). `robusto` = enable-check vs los 3 compañeros.

### Día de competencia (2026-07-13, todos M3 8M: BC warm-start + solo_heavy salvo indicado)

| Layout | seed | enfoque | enable vs greedy | robusto? | nota |
|---|---|---|---|---|---|
| asymmetric (E1) | 510 | M3 solo_heavy | 60487 vs 65428 | ❌ | el más cercano en E1 |
| asymmetric (E1) | 511 | M3 solo_heavy | 50807 vs 65428 | ❌ | |
| asymmetric (E1) | 512 | M3 solo_heavy | 60456 vs 65428 | ❌ | (packed pregrado) |
| coordination (E2) | 520 | M3 solo_heavy | 50230 vs 80244 | ❌ | planner 8s imbatible |
| coordination (E2) | 521 | M3 **sticky_heavy** | 60460 vs 80244 | ❌ | vs sticky: 3.3s < planner 3.8s |
| coordination (E2) | 522 | M3 **sticky_heavy** | 60478 vs 80244 | ❌ | |
| counter_circuit (E3) | 500 | M3 solo_heavy | 81088 vs 136154 | ❌ | 8.1s — planner receta 13.5s |
| counter_circuit (E3) | 501 | M3 solo_heavy | 80612 vs 136154 | ❌ | best ckpt tocó ~10s en train |
| cramped_room | 530 | M3 solo_heavy | 55445 vs 65298 | ❌ | **gana eps** (39704 vs 28540) |
| cramped_room | 531 | M3 solo_heavy | 60466 vs 65298 | ❌ | **gana eps** (48524 vs 28540) |
| counter_circuit_o_1order | 540 | M3 solo_heavy | 40406 vs 60256 | ❌ | seguro por nombre de E3 |

### Históricos (pre-competencia)

| Layout | seed | enfoque | eval(vs greedy) | robusto? | nota |
|---|---|---|---|---|---|
| cramped_room | 0/1 | population | 40537-40550 | — | 1ª ronda (A100) |
| cramped_room | 100/101 | greedy | ~50400 | ❌ | gana no-swap, empata swap |
| cramped_room | 102 | greedy_heavy | 46975 | ❌ | pierde greedy, 0 vs random |
| asymmetric | 0 / 100 | population / greedy | 33527 / 47049 | ❌ | pierde hasta vs greedy |
| coordination_ring | 100 | greedy | 40556 | ❌ | planner 8s, imbatible |
| custom_room | 100/101/300/301 | varios | 46961-50422 | ❌ | 0 vs random o pierde eps |
| custom_dual_pots | 100/200/300 | varios | 60193-60377 | ❌ | idem |
| **custom_zigzag** | **300** | **M3 BC+solo_heavy** | **50412** | **✅ HABILITADO** | **70286/47410/= → G8 PASS** |
| custom_zigzag | 200/301 | solo_heavy / M4 self-play | 50273/60443 | ✅* | robustos, < campeón → respaldo |
| custom_zigzag | 100/302 | greedy_heavy / curriculum | ~50280 | ❌ | |
| **rehearsal_kitchen** | **400** | **M3 BC (ensayo playbook)** | 40331 | **✅ HABILITADO** | **+376% vs planner; playbook validado 291 min** |
| rehearsal_kitchen | 401 | M3 BC | 20634 | — | no evaluado (400 ya robusto) |

*✅ con la sonda de cooperación activa (default del StudentAgent desde 2026-07-11).

---

## 5. Enable-check detallado — layouts de competencia (2026-07-13)

✅ = student (PPO+sonda) ≥ planner. Se habilita solo si las **3** columnas son ✅.
La columna random empata SIEMPRE (la sonda manda al planner) — el muro nuevo es greedy/eps.

| Modelo | vs greedy | vs greedy_eps | vs random | ¿robusto? |
|---|---|---|---|---|
| asymmetric s510 | 60487 ❌ (65428) | 55441 ❌ (73448) | = ✅ | NO |
| coordination s521 (sticky_heavy) | 60460 ❌ (80244) | 38638 ❌ (48682) | = ✅ | NO |
| counter_circuit s500 | 81088 ❌ (136154) | 48514 ❌ (123545) | = ✅ | NO |
| cramped s531 | 60466 ❌ (65298) | **48524 ✅ (28540)** | = ✅ | NO — solo falla greedy |

(resto de seeds en `outputs/dayof/*/enable_seed*.json`; patrón idéntico)

**Dato extra**: el especialista sticky (s521) medido contra su propio objetivo con
`check_vs_sticky --model-path`: **3.3 sopas vs sticky < planner 3.8** — ni en su celda gana.

---

## 6. Los dos muros (histórico + hoy)

- **Muro 1 — "0 vs random"** (layouts custom, resuelto por construcción): el PPO
  entrenado con greedy pierde la habilidad solo. Solución: **sonda de cooperación**
  (selector), no fuerza bruta. Sigue vigente y activa.
- **Muro 2 — "el piso del planner"** (E1-E3, hoy): con la sonda, el student ya nunca
  pierde vs random; pero el planner mejorado (modo receta, anti-oscilación, cocinas
  disjuntas) rinde 6.5-13.5 sopas y los PPO de 8M se quedan a 60-80% de eso vs greedy.
  No es un fallo: significa que **la entrega fuerte ES el planner** y el PPO solo debe
  habilitarse donde demuestre superarlo (zigzag y rehearsal lo demuestran; E1-E3 no).

---

## 7. Estado actual

**Sin jobs activos.** Los 11 modelos del 2026-07-13 están entrenados y evaluados
(2 tandas: 5 jobs playbook + 2 jobs packed con 6 entrenamientos). Cola limpia.

---

## 8. Próximos pasos

1. **Lunes temprano (E4)**: `python -m scripts.planner_baseline --layout <e4>` (o
   `--layout-file` si es custom). Con compañero random_motion el planner es el plan A
   (hace todo el ciclo solo); playbook disponible si hay horas y el PPO promete.
2. **E5/E6 (en vivo)**: planner universal + fallback por terrain-hash (si el layout
   coincide con zigzag/rehearsal/otros habilitados, el PPO se activa solo).
3. **Export `agent.pt`**: pendiente la spec del profesor (¿TorchScript de la red /
   state_dict / pickle?). Ver conversación — 3 preguntas abiertas al profesor.

## 9. Cómo reproducir

```bash
source scripts/env.sh
# baseline del planner en un layout builtin o custom:
python -m scripts.planner_baseline --layout counter_circuit
# eval vs compañeros sticky (E2/E3), planner o candidato:
python -m scripts.check_vs_sticky --layout coordination_ring --agent planner
python -m scripts.check_vs_sticky --layout coordination_ring --agent student \
    --model-path models/coordination_ring/seed521/best.zip
# playbook día-de-competencia (builtin o .layout):
bash scripts/prepare_new_layout.sh counter_circuit 2 8000000
# entrenamientos EMPAQUETADOS (N por job — esquiva bug epilog y cuota de 5 jobs):
sbatch --export=ALL,JOBS=training/jobs_packed_tesis.txt sbatch/train/run_train_packed.sh
# evaluar un candidato (robusto, 3 compañeros):
python -m scripts.enable_model --layout counter_circuit --model models/counter_circuit/seed500/best.zip
# entrega integrada:
python -m evaluation.run_gate --gate G8
```
