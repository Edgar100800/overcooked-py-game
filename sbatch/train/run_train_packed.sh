#!/bin/bash
#SBATCH --job-name=ovppo-pack
#SBATCH --partition=standard
#SBATCH --account=tesis
#SBATCH --qos=a-tesis
#SBATCH --cpus-per-task=30
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ppo-pack-%j.out
#SBATCH --error=logs/ppo-pack-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luccianachambillaugc@gmail.com

# Job EMPAQUETADO: corre TODAS las lineas del manifiesto en PARALELO dentro de un
# solo job SLURM. Por que: el bug del epilog (docs/CLUSTER_NOTES.md §2) prohibe
# tener 2+ jobs propios en el mismo nodo, y la cuota QOS limita los JOBS (3+2) y
# los CPU (32/cuenta) -- pero un job de 30 cpus puede alojar 3 entrenamientos de
# ~10 hilos cada uno sin co-ubicacion de jobs. Asi la cuota de 5 jobs rinde hasta
# ~6 entrenamientos concurrentes en lugar de 5.
#
# Uso:
#   sbatch --export=ALL,JOBS=training/jobs_packed_X.txt sbatch/train/run_train_packed.sh
# Cada linea del manifiesto: layout | layout_file("-"=builtin) | seed | steps | extra

set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
if [ -f /etc/profile.d/z00_lmod.sh ]; then source /etc/profile.d/z00_lmod.sh; fi
module load python3/3.10.2 2>/dev/null || true

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
JOBS=${JOBS:-training/jobs.txt}
N_ENVS=${N_ENVS:-8}
DEVICE=${DEVICE:-cpu}

echo "[packed] node=$SLURMD_NODENAME cpus=$SLURM_CPUS_PER_TASK manifest=$JOBS"

PIDS=(); NAMES=()
while IFS= read -r line; do
  IFS='|' read -r LAYOUT LAYOUT_FILE SEED TIMESTEPS EXTRA <<< "$line"
  LAYOUT=$(echo "$LAYOUT" | xargs); LAYOUT_FILE=$(echo "$LAYOUT_FILE" | xargs)
  SEED=$(echo "$SEED" | xargs);     TIMESTEPS=$(echo "$TIMESTEPS" | xargs)
  EXTRA=$(echo "$EXTRA" | xargs)

  if [ "$LAYOUT_FILE" = "-" ] || [ -z "$LAYOUT_FILE" ]; then
    LAYOUT_ARG=(--layout "$LAYOUT"); KEY="$LAYOUT"
  else
    LAYOUT_ARG=(--layout-file "$LAYOUT_FILE"); KEY=$(basename "$LAYOUT_FILE" .layout)
  fi
  OUT="models/${KEY}/seed${SEED}"
  LOG="logs/ppo-pack-${SLURM_JOB_ID}-${KEY}-seed${SEED}.log"

  echo "[packed] lanzo $KEY seed=$SEED steps=$TIMESTEPS -> $OUT (log: $LOG)"
  $PYTHON_BIN -m training.train_ppo \
    "${LAYOUT_ARG[@]}" --seed "$SEED" --timesteps "$TIMESTEPS" --out "$OUT" \
    --device "$DEVICE" --obs lossless_grid --n-envs "$N_ENVS" $EXTRA \
    > "$LOG" 2>&1 &
  PIDS+=($!); NAMES+=("$KEY/seed$SEED")
done < <(grep -vE '^\s*#|^\s*$' "$JOBS")

FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[packed] OK   ${NAMES[$i]}"
  else
    echo "[packed] FAIL ${NAMES[$i]} (ver su log)"
    FAIL=1
  fi
done
exit $FAIL
