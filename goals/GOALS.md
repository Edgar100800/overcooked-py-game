# GOALS.md — Gates medibles (copiado de PLAN.md §11)

Config de gate salvo indicación: horizon 250, `max_action_time_ms: 100`, 5 episodios
con `gate_seeds` promediando; "swap" = correr además con `swap_agent_positions: true`.
Un gate solo lo aprueba `evaluation/run_gate.py` con rollouts reales. `progress.json`
lo escribe SOLO ese script.

| Gate | Prueba de concepto | Criterio (medido por run_gate.py) |
|---|---|---|
| **G0** | Smoke del entorno | stay vs greedy en cramped_room sin excepciones; ≥1 sopa del greedy en el step log |
| **G1** | Score oficial correcto | `official_score` reproduce 3 casos sintéticos Y `verify.py` == reporte en un rollout real. Al pasar → **freeze de evaluation/** |
| **G2** | Planner mueve e interactúa | Planner vs `stay` en cramped_room: ≥1 sopa promedio, 0 timeouts, p99 < 50 ms |
| **G3** | Planner competitivo | Planner vs `greedy_full_task` en 6 layouts: ≥2 sopas promedio/layout, 0 timeouts, con y sin swap |
| **G4** | Planner autosuficiente | Planner vs `random_motion` en 6 layouts: ≥1 sopa promedio/layout, con y sin swap |
| **G5** | Gym env válido | `check_env` de SB3 pasa; PPO 50k sin NaN y shaped sube vs aleatorio |
| **G6** | PPO aprende (smoke) | PPO en cramped_room vs greedy, ≥2M steps: score oficial > 0 en gate_seeds |
| **G7** | PPO supera al planner | En ≥1 layout: score PPO > score planner (mismos seeds/compañero), p99 < 50 ms, con y sin swap. Si no → selector queda en planner (decisión válida, documentar) |
| **G8** | Entrega integrada | `student_agent.py` en 6 layouts × {greedy, greedy+ε, random_motion} × swap: 0 crashes/inválidas/timeouts, score ≥ planner puro en cada celda |

**Avanzar** = nº de gates aprobados en `progress.json`. Meta mínima noche: G0–G4 + G8
solo-planner. Meta ideal: G0–G8.

Comando: `python -m evaluation.run_gate --gate GX`
