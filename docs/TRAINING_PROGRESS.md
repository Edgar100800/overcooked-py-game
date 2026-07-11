# Avance de entrenamientos PPO — Overcooked

Estado consolidado de todos los entrenamientos y sus resultados. Actualizado 2026-07-11.

> **Verdad = env.** Todos los scores son con el score oficial de la competencia
> (`10000·sopas + timing − timeouts`) en `gate_seeds`, con y sin swap. `~Ns` = nº de sopas.
> El enable-check (`scripts/enable_model.py`) compara el PPO vs el planner puro contra los
> **3 compañeros** que prueba G8 (greedy, greedy_eps, random_motion). Se **habilita** un
> modelo solo si NO empeora al planner en NINGÚN compañero.

---

## 1. Resumen ejecutivo

- **Entrega actual:** el selector usa el **planner robusto** en todos los layouts (G8 PASS,
  garantizado). Ningún PPO habilitado todavía.
- **Lo que aprende el PPO:** con compañero greedy, el PPO iguala o **supera** al planner en
  varios layouts (hasta 8 sopas). Con `greedy_eps` suele ganar también.
- **El muro:** vs `random_motion` **todos** los modelos colapsan a **0 sopas** → rompen la
  garantía "nunca peor" de G8 → no se habilitan. Es un problema de *zero-shot coordination*:
  el PPO aprendió a depender del compañero y no domina el ciclo en solitario.
- **En curso:** 2 modelos con enfoque **menos greedy-dependiente** (solo_heavy, curriculum)
  que atacan justo ese muro.

---

## 2. La barra a superar — planner puro (baseline)

| Layout | vs greedy | vs random | ← el PEOR de estos dos es lo que el PPO debe superar |
|---|---|---|---|
| coordination_ring | 80244 (8.0s) | 26737 (2.6s) | random débil, greedy fortísimo |
| asymmetric_advantages | 65428 (6.5s) | 50690 (5.0s) | fuerte en ambos |
| cramped_room | 65298 (6.5s) | 50386 (5.0s) | fuerte en ambos |
| custom_room | 60210 (6.0s) | 50389 (5.0s) | medio |
| custom_dual_pots | 56073 (5.5s) | 60570 (6.0s) | medio |
| **custom_zigzag_kitchen** | **36287 (3.5s)** | 50425 (5.0s) | **planner DÉBIL vs greedy → mejor blanco** |

---

## 3. Enfoques probados (compañeros de entrenamiento)

| Enfoque | Mezcla de compañeros | Idea |
|---|---|---|
| **population** (default) | greedy 35 / sticky+eps 25 / random 20 / self-play 15 / stay 5 | balanceado |
| **greedy (100%)** | greedy 100 | especializar (sobreajusta) |
| **greedy_heavy** | greedy 55 / sticky+eps 15 / random 25 / stay 5 | fuerte vs greedy, algo robusto |
| **solo_heavy** ⏳ | stay 25 / random 35 / greedy 30 / sticky+eps 10 | **60% no-coop → fuerza solo** |
| **curriculum** ⏳ | fase-1: stay/random 50/50 → fase-2: solo_heavy | **aprende solo primero, luego coopera** |

---

## 4. Tabla maestra de modelos

`eval` = score proxy vs greedy (índice aleatorio) del mejor checkpoint. `robusto` = resultado
del enable-check vs los 3 compañeros.

| Layout | seed | enfoque | steps | eval(vs greedy) | robusto? | nota |
|---|---|---|---|---|---|---|
| cramped_room | 0 | population | 5M | 40537 | — | 1ª ronda (A100) |
| cramped_room | 1 | population | 4.8M | 40550 | — | |
| cramped_room | 100 | greedy | 5M | 50458 | ❌ | gana no-swap, empata swap |
| cramped_room | 101 | greedy | 5M | 50380 | ❌ | idem |
| cramped_room | 102 | greedy_heavy | 8M | 46975 | ❌ | pierde greedy, 0 vs random |
| asymmetric | 0 | population | 4.6M | 33527 | — | |
| asymmetric | 100 | greedy | 5M | 47049 | ❌ | pierde hasta vs greedy |
| coordination_ring | 100 | greedy | 5M | 40556 | ❌ | planner 8s, imbatible |
| custom_room | 100 | greedy | 8M | 50422 | ❌* | *G7 lo habilitó, rompió G8 vs random |
| custom_room | 101 | greedy_heavy | 8M | 50177 | ❌ | gana greedy+eps, **solo 0 vs random** |
| custom_dual_pots | 100 | greedy_heavy | 8M | 60377 | ❌ | gana greedy(8s!), 0 vs random |
| **custom_zigzag** | 100 | greedy_heavy | 8M | 50289 | ❌ | **gana greedy+eps, solo 0 vs random** |
| custom_zigzag | 200 | **solo_heavy** | 8M | ⏳ | ⏳ | corriendo (n004) |
| custom_dual_pots | 200 | **curriculum** | 8M | ⏳ | ⏳ | corriendo (n005) |

---

## 5. Enable-check detallado (PPO vs planner por compañero)

✅ = PPO ≥ planner. El modelo se habilita solo si las **3** columnas son ✅.

| Layout (enfoque) | vs greedy | vs greedy_eps | vs random_motion | ¿robusto? |
|---|---|---|---|---|
| custom_room (greedy) | 70292 ✅ | 36827 ✅ | **0** ❌ (50389) | NO |
| custom_room (greedy_heavy) | 60326 ✅ | 41578 ✅ | **0** ❌ (50389) | NO |
| **custom_zigzag (greedy_heavy)** | 60441 ✅ | 51491 ✅ | **0** ❌ (50425) | **NO — solo falta random** |
| custom_dual_pots (greedy_heavy) | 80326 ✅ | 56428 ❌ | **0** ❌ | NO |
| cramped (greedy_heavy) | 60321 ❌ | 42776 ✅ | **0** ❌ | NO |
| asymmetric (greedy) | 60306 ❌ | 36845 ❌ | **0** ❌ | NO |

**Patrón clarísimo:** la columna `random_motion` es **0 en TODOS** los modelos. En
`custom_room` y `custom_zigzag` es lo ÚNICO que falla (ganan vs los 2 compañeros greedy).

---

## 6. El hallazgo clave — el muro del "0 vs random"

- **Qué:** vs un compañero que no coopera (`random_motion`, nunca interactúa), el PPO hace
  **exactamente 0 sopas**. No es "poco": es cero, en 2500 steps de evaluación.
- **Por qué:** entrenando mayormente con greedy, el PPO aprende una **división de trabajo**
  (deja que el compañero haga parte del ciclo). Cuando el compañero es inútil, los estados
  se salen de distribución y la política determinista entra en un **bucle inútil** → 0 sopas.
- **El planner no sufre esto** porque está codificado a mano para completar el ciclo solo
  (5 sopas vs random).
- **La solución (en curso):** entrenar con mayoría de compañeros no-cooperativos (solo_heavy)
  o un **curriculum solo-primero**, para que el PPO domine el ciclo en solitario ANTES de
  aprender a cooperar. Objetivo medible: que la columna `random_motion` deje de ser 0 y
  supere el score del planner (p.ej. ≥5 sopas en zigzag).

---

## 7. Mejor candidato a victoria robusta

**`custom_zigzag_kitchen`**: el planner es débil ahí (3.5 sopas vs greedy). El PPO greedy_heavy
ya **le gana vs greedy (60441 vs 36287) y vs greedy_eps (51491 vs 41879)**. Lo único que falta
es superar el 5.0 del planner vs random. Si solo_heavy/curriculum rompe el 0-vs-random ahí,
custom_zigzag pasa a ser una **victoria robusta** → G8 con PPO activo.

---

## 8. Estado actual (corriendo)

| Job | Cuenta/Nodo | Layout | enfoque | progreso |
|---|---|---|---|---|
| 46065 | pregrado/n003 | custom_zigzag | greedy_heavy | ~7.5M/8M (terminando) |
| 46096 | tesis/n004 | custom_zigzag | **solo_heavy** | recién iniciado |
| 46097 | pregrado/n005 | custom_dual_pots | **curriculum** | recién iniciado (fase-1) |

Infra: CPU, un job por nodo (evita el bug del epilog, ver `docs/CLUSTER_NOTES.md`). ~2.2h.

---

## 9. Próximos pasos

1. Al terminar los 2 nuevos (solo_heavy, curriculum): `enable_model.py` → **¿random deja
   de ser 0?**
2. Si sí y supera al planner en zigzag → habilitar → `run_gate G8` con PPO activo = **meta
   cumplida** (barra completa).
3. Si mejora pero no alcanza: subir no-coop a 70% o alargar la fase-1 del curriculum, repetir.
4. Alternativa ya disponible: despliegue **por-escenario** (PPO para escenarios 1–3 con
   compañero greedy, donde ya gana; planner para 4–6).

## 10. Cómo reproducir

```bash
source scripts/env.sh
# entrenar (CPU, un job por nodo):
sbatch --array=0-0 --nodelist=n004 --export=ALL,JOBS=training/jobs_stepA5.txt sbatch/train/run_train_ppo.sh
# evaluar un candidato (robusto, 3 compañeros):
python -m scripts.enable_model --layout custom_zigzag_kitchen \
    --layout-file configs/layouts/custom_zigzag_kitchen.layout \
    --model models/custom_zigzag_kitchen/seed200/best.zip
# entrega integrada:
python -m evaluation.run_gate --gate G8
```
