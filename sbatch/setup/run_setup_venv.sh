#!/bin/bash
#SBATCH --job-name=ovsetup
#SBATCH --partition=standard
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

# Construye el .venv del proyecto en un nodo del cluster (una sola vez).
# Tambien se puede correr directo en el login: bash scripts/setup_venv.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
bash scripts/setup_venv.sh
