#!/bin/bash
# night_loop.sh — orquestador de la corrida nocturna (PLAN.md §8-9).
#
# Todos los gates ya estan codificados, asi que el loop es mecanico: corre cada gate
# con run_gate.py (unico que escribe progress.json), registra en PROGRESS.md y hace
# commit. Los entrenamientos largos van a la A100 via sbatch (job array); cuando hay
# modelos, corre G6/G7 (habilita el mejor) y re-evalua G8.
#
# Uso:
#   nohup bash scripts/night_loop.sh > logs/night_loop.log 2>&1 &
#   FASE=cpu bash scripts/night_loop.sh     # solo gates CPU (G0..G5,G8) + freeze G1
#   FASE=train bash scripts/night_loop.sh   # submitea entrenamiento A100
#   FASE=ppo  bash scripts/night_loop.sh    # G6/G7 por layout + G8 (tras entrenar)
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1
PY=.venv/bin/python
FASE="${FASE:-all}"

run_gate() {  # $1 = gate, resto = args extra
  local g="$1"; shift || true
  echo "=== $(date '+%F %T') run_gate $g $* ==="
  if $PY -m evaluation.run_gate --gate "$g" "$@" 2>&1 | grep -vE "Gym has been|Please upgrade|Users of this|migration guide" | tail -40; then
    :
  fi
  git add -A goals/ models/ 2>/dev/null || true
  local status; status=$($PY -c "import json;p=json.load(open('goals/progress.json'));print('PASS' if p.get('$g',{}).get('passed') else 'FAIL')" 2>/dev/null || echo "UNKNOWN")
  git commit -q -m "gate $g: $status" 2>/dev/null || echo "(sin cambios que commitear)"
  echo "=== $g -> $status ==="
}

fase_cpu() {
  # Gates que no necesitan GPU. G1 (freeze) al final, tras estabilizar el resto.
  for g in G0 G2 G3 G4 G5; do run_gate "$g"; done
  run_gate G8                 # selector en modo solo-planner (entrega garantizada)
  run_gate G1                 # <-- congela evaluation/ (debe ir tras validar todo)
}

fase_train() {
  echo "=== submit entrenamiento A100 (job array) ==="
  sbatch sbatch/train/run_train_ppo.sh || echo "sbatch fallo (sin SLURM?)"
  echo "Monitoreo: squeue -u \$USER ; tail -f logs/ppo-*.out"
}

fase_ppo() {
  # Para cada layout con best.zip: G6 (aprende) y G7 (supera al planner -> habilita).
  for d in models/*/; do
    [ -f "$d/seed0/best.zip" ] || [ -f "$d/best.zip" ] || continue
    local key; key=$(basename "$d")
    local model; model=$(ls "$d"/seed*/best.zip "$d"/best.zip 2>/dev/null | head -1)
    run_gate G6 --layout "$key" --model "$model"
    run_gate G7 --layout "$key" --model "$model"
  done
  run_gate G8                 # re-evaluar con los modelos habilitados
}

case "$FASE" in
  cpu)   fase_cpu ;;
  train) fase_train ;;
  ppo)   fase_ppo ;;
  all)   fase_cpu; fase_train ;;
  *) echo "FASE desconocida: $FASE"; exit 1 ;;
esac
echo "=== night_loop ($FASE) terminado $(date '+%F %T') ==="
