#!/bin/bash
#SBATCH --job-name=ovgate
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/gate-%j.out
#SBATCH --error=logs/gate-%j.err

# Corre un gate en CPU (los gates se validan SIEMPRE en el env oficial, CPU; PLAN §10).
# Uso: sbatch sbatch/eval/run_gate.sh G3
#      sbatch sbatch/eval/run_gate.sh G8
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
if [ -f /etc/profile.d/z00_lmod.sh ]; then source /etc/profile.d/z00_lmod.sh; fi
module load python3/3.10.2 2>/dev/null || true

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
GATE=${1:?uso: sbatch sbatch/eval/run_gate.sh GX}
shift || true

echo "[run_gate.sh] gate=$GATE args=$*"
$PYTHON_BIN -m evaluation.run_gate --gate "$GATE" "$@"
