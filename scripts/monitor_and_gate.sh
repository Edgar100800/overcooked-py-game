#!/bin/bash
# Espera al array de entrenamiento $JID y luego corre los gates PPO (FASE=ppo).
set -eo pipefail
cd "$(dirname "$0")/.."
JID="${JID:?uso: JID=<jobid> bash scripts/monitor_and_gate.sh}"
echo "[monitor] esperando array $JID ..."
iters=0
while squeue -j "$JID" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING"; do
  echo "[monitor] $(date '+%F %T')"; squeue -j "$JID" -h -o "  %.14i %.8T %R" 2>/dev/null | head
  iters=$((iters+1)); [ "$iters" -gt 200 ] && { echo "[monitor] timeout"; break; }
  sleep 180
done
echo "[monitor] entrenamiento terminado -> gates PPO"
FASE=ppo bash scripts/night_loop.sh
