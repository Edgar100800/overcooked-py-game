#!/bin/bash
#SBATCH --job-name=ovppo
#SBATCH --partition=standard
#SBATCH --account=tesis
#SBATCH --qos=a-tesis
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ppo-%A_%a.out
#SBATCH --error=logs/ppo-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luccianachambillaugc@gmail.com

# Job array de entrenamiento PPO en CPU (nodos n[003-006], particion standard).
# Cada tarea del array = una linea de training/jobs.txt = un (layout, seed, config).
#
# POR QUE CPU (no A100): Overcooked es CPU-bound (PLAN §16) y el CNN es diminuto, asi
# que la GPU casi no acelera. Ademas, correr varios jobs concurrentes sobre UNA A100
# compartida por shards/MPS es fragil: cuando un job co-ubicado termina, el epilog de
# limpieza del nodo tira el servidor MPS y MATA (SIGKILL) a los otros jobs (fue lo que
# paso con el job 46045; ver ADDENDUM del plan). CPU elimina esa clase de fallo.
#
# Uso:
#   sbatch sbatch/train/run_train_ppo.sh                 # array 0-2 (3 jobs, jobs.txt)
#   sbatch --array=0-2 -N1 sbatch/train/run_train_ppo.sh
# Limites QOS a-tesis: MaxSubmit=5, MaxJobs=3 corriendo, cpu<=32. Por eso array 0-2 y
# cpus-per-task=10 (3x10=30<=32). Mas layouts -> 2da tanda (--array=3-6).
#
# ALTERNATIVA A100 (si se insiste en GPU): usar rebanadas MIG DEDICADAS (aisladas, sin
# MPS) y limitar concurrencia a las 2 que existen:
#   sbatch --partition=gpu --gres=gpu:a100_1g.5gb:1 --array=0-6%2 \
#          --export=ALL,DEVICE=cuda sbatch/train/run_train_ppo.sh
# (el %2 evita el fallo MPS; DEVICE=cuda cambia el device abajo).

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Modulos (el .venv se creo con python3/3.10.2; necesita su libpython en runtime).
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
if [ -f /etc/profile.d/z00_lmod.sh ]; then source /etc/profile.d/z00_lmod.sh; fi
module load python3/3.10.2 2>/dev/null || true

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1          # 1 hilo por worker (CPU-bound; evita sobre-suscripcion)
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
JOBS=${JOBS:-training/jobs.txt}
N_ENVS=${N_ENVS:-8}
DEVICE=${DEVICE:-cpu}             # cpu por defecto; DEVICE=cuda para la alternativa A100
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# N-esima linea NO comentada / no vacia del manifiesto.
line=$(grep -vE '^\s*#|^\s*$' "$JOBS" | sed -n "$((TASK_ID + 1))p")
if [ -z "$line" ]; then echo "No hay job para TASK_ID=$TASK_ID en $JOBS"; exit 1; fi

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

echo "[run_train_ppo] task=$TASK_ID layout=$LAYOUT file=$LAYOUT_FILE seed=$SEED steps=$TIMESTEPS out=$OUT device=$DEVICE"
echo "[run_train_ppo] node=$SLURMD_NODENAME cpus=$SLURM_CPUS_PER_TASK gres=${SLURM_JOB_GRES:-none}"
[ "$DEVICE" = "cuda" ] && { module load cuda/11.8 2>/dev/null || true; nvidia-smi -L 2>/dev/null || true; }

$PYTHON_BIN -m training.train_ppo \
  "${LAYOUT_ARG[@]}" --seed "$SEED" --timesteps "$TIMESTEPS" --out "$OUT" \
  --device "$DEVICE" --obs lossless_grid --n-envs "$N_ENVS" $EXTRA

echo "[run_train_ppo] done -> $OUT (best.zip por score oficial)"
