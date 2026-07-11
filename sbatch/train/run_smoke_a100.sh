#!/bin/bash
#SBATCH --job-name=ovsmoke
#SBATCH --partition=gpu
#SBATCH --account=tesis
#SBATCH --qos=a-tesis
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

# Smoke de entrenamiento en la A100 (ag001): valida que el camino GPU/MIG/cuda
# funciona end-to-end y produce un best.zip. ~150k steps (rapido).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
if [ -f /etc/profile.d/z00_lmod.sh ]; then source /etc/profile.d/z00_lmod.sh; fi
module load python3/3.10.2 cuda/11.8 2>/dev/null || module load python3/3.10.2
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
PY=.venv/bin/python

echo "[smoke] node=$SLURMD_NODENAME gres=$SLURM_JOB_GRES"
nvidia-smi -L || true
$PY -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

$PY -m training.train_ppo --layout cramped_room --seed 0 --timesteps 150000 \
  --out models/cramped_room/smoke_a100 --device cuda --obs lossless_grid \
  --n-envs 6 --eval-freq 50000
echo "[smoke] done -> models/cramped_room/smoke_a100"
