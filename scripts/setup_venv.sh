#!/bin/bash
# setup_venv.sh — construye el entorno virtual reproducible para overcooked-py-game
# en el cluster khipu. Ejecutable en el nodo de login o via sbatch/setup/run_setup_venv.sh.
#
# Uso:
#   bash scripts/setup_venv.sh            # crea .venv con python3/3.10.2 (recomendado)
#   PYVER=3.8.0 bash scripts/setup_venv.sh  # fallback si overcooked-ai no compila en 3.10
#
# Notas de infraestructura (khipu):
#   - venv normal aisla de ~/.local por defecto -> evita el bug pip->~/.local de conda.
#   - torch se instala con el indice cu118 para correr en la A100 (ag001). El mismo
#     wheel funciona en CPU (login / gates).
#   - overcooked-ai fija numpy<2 (ver requirements.txt); no actualizar numpy.
set -eo pipefail
# Nota: NO usar 'set -u' global; los scripts de Lmod de khipu referencian
# variables sin definir (p.ej. SLURM_NODELIST) y abortarian el sourcing.

cd "$(dirname "$0")/.."   # raiz del repo
REPO_ROOT="$(pwd)"
PYVER="${PYVER:-3.10.2}"
CUDAVER="${CUDAVER:-11.8}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[setup_venv] repo: $REPO_ROOT"
echo "[setup_venv] python module: python3/$PYVER  cuda: cuda/$CUDAVER  venv: $VENV_DIR"

# Cargar Lmod en shells no interactivos (patron khipu) y los modulos.
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
if [ -f /etc/profile.d/z00_lmod.sh ]; then source /etc/profile.d/z00_lmod.sh; fi
module load "python3/$PYVER" "cuda/$CUDAVER" 2>/dev/null || module load "python3/$PYVER"

# Evitar contaminacion desde ~/.local
export PYTHONNOUSERSITE=1

python3 --version
python3 -m venv "$VENV_DIR"
PY="$REPO_ROOT/$VENV_DIR/bin/python"

"$PY" -m pip install -U pip wheel setuptools

# torch con CUDA 11.8 (compatible con la A100). Si falla la red del indice cuda,
# reintentar sin --index-url (cae al wheel CPU/cuda por defecto de PyPI).
echo "[setup_venv] instalando torch (cu118)..."
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu118 \
  || "$PY" -m pip install torch

echo "[setup_venv] instalando overcooked-ai y dependencias base..."
"$PY" -m pip install -r requirements.txt

echo "[setup_venv] instalando dependencias RL..."
"$PY" -m pip install -r requirements-rl.txt

echo "[setup_venv] verificando imports..."
"$PY" - <<'PYCODE'
import numpy, overcooked_ai_py, stable_baselines3, gymnasium, torch
print("numpy      ", numpy.__version__)
print("overcooked ", overcooked_ai_py.__version__ if hasattr(overcooked_ai_py, "__version__") else "ok")
print("sb3        ", stable_baselines3.__version__)
print("gymnasium  ", gymnasium.__version__)
print("torch      ", torch.__version__, "| cuda build:", torch.version.cuda)
assert numpy.__version__.startswith("1."), "numpy debe ser <2 (overcooked-ai)"
print("[setup_venv] OK")
PYCODE

echo "[setup_venv] listo. Activar con: source $VENV_DIR/bin/activate"
