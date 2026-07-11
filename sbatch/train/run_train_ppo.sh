#!/bin/bash
#SBATCH --job-name=ovppo
#SBATCH --partition=gpu
#SBATCH --account=tesis
#SBATCH --qos=a-tesis
#SBATCH --gres=shard:a100_1g.5gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ppo-%A_%a.out
#SBATCH --error=logs/ppo-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luccianachambillaugc@gmail.com

# Job array de entrenamiento PPO en la A100 (nodo ag001, MIG expuesto por SLURM).
# Cada tarea del array = una linea de training/jobs.txt = un (layout, seed, config).
#
# Uso:
#   sbatch sbatch/train/run_train_ppo.sh                      # array 0-6 (jobs.txt)
#   sbatch --array=0-1 sbatch/train/run_train_ppo.sh          # solo 2 tareas
#   # Mayor concurrencia (CPU-bound): usar shards MPS en vez de rebanadas MIG dedicadas
#   # (hay 2x a100_1g.5gb pero 32 shards). Overcooked casi no usa la GPU:
#   sbatch --gres=shard:a100_1g.5gb:1 sbatch/train/run_train_ppo.sh
#
# Limites del QOS a-tesis (sacctmgr show qos a-tesis): MaxSubmit=5, MaxJobs=3 (corriendo),
# cpu<=32 total. Por eso: array 0-2 (3 jobs), cpus-per-task=10 (3x10=30<=32), shards MPS
# (2 rebanadas MIG 1g.5gb dedicadas no alcanzan para 3 jobs; 32 shards si). Para mas
# layouts, correr en tandas (jobs.txt tiene 7 lineas; --array=3-5 en una 2da tanda).

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Modulos (el .venv se creo con python3/3.10.2; necesita su libpython en runtime).
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
if [ -f /etc/profile.d/z00_lmod.sh ]; then source /etc/profile.d/z00_lmod.sh; fi
module load python3/3.10.2 cuda/11.8 2>/dev/null || module load python3/3.10.2

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1          # 1 hilo por worker (CPU-bound; evita sobre-suscripcion)
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
JOBS=${JOBS:-training/jobs.txt}
N_ENVS=${N_ENVS:-10}
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

echo "[run_train_ppo] task=$TASK_ID layout=$LAYOUT file=$LAYOUT_FILE seed=$SEED steps=$TIMESTEPS out=$OUT"
echo "[run_train_ppo] gres=$SLURM_JOB_GRES node=$SLURMD_NODENAME cpus=$SLURM_CPUS_PER_TASK"
nvidia-smi -L 2>/dev/null || echo "(sin nvidia-smi visible)"

$PYTHON_BIN -m training.train_ppo \
  "${LAYOUT_ARG[@]}" --seed "$SEED" --timesteps "$TIMESTEPS" --out "$OUT" \
  --device cuda --obs lossless_grid --n-envs "$N_ENVS" $EXTRA

echo "[run_train_ppo] done -> $OUT (best.zip por score oficial)"
