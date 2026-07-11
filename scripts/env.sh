# env.sh — sourcear para usar el .venv en khipu de forma interactiva.
#   source scripts/env.sh
# Carga los modulos (necesario: el .venv se creo con python3/3.10.2 y su
# libpython vive en el modulo -> sin el modulo, .venv/bin/python no arranca) y
# activa el entorno virtual.
PYVER="${PYVER:-3.10.2}"
CUDAVER="${CUDAVER:-11.8}"
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
if [ -f /etc/profile.d/z00_lmod.sh ]; then source /etc/profile.d/z00_lmod.sh; fi
module load "python3/$PYVER" "cuda/$CUDAVER" 2>/dev/null || module load "python3/$PYVER"
export PYTHONNOUSERSITE=1
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
# shellcheck disable=SC1091
source "$_here/.venv/bin/activate"
echo "[env] venv activo: $(which python)  ($(python --version 2>&1))"
