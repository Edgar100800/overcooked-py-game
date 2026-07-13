#!/bin/bash
# prepare_new_layout.sh — playbook DIA-DE-COMPETENCIA: .layout nuevo -> PPO habilitado.
#
# Convierte un layout recien revelado en un modelo PPO robusto y habilitado usando la
# receta ganadora M3 (BC warm-start del planner + PPO solo_heavy + sonda de cooperacion).
# Fases: baseline planner -> dataset BC -> entrenamientos SLURM (1 job por nodo, ver
# docs/CLUSTER_NOTES.md) -> enable-check robusto -> reporte con tiempos.
#
# Uso:
#   bash scripts/prepare_new_layout.sh <layout> [SEEDS] [STEPS]
#     <layout>: ruta a un .layout custom (configs/layouts/x.layout) O nombre de un
#               layout builtin de overcooked_ai_py (ej. asymmetric_advantages)
#     SEEDS: cuantos seeds entrenar en paralelo (default 2; seeds 400,401,...)
#     STEPS: timesteps por entrenamiento (default 8000000 ~ 3.3h en CPU)
# Todo queda en outputs/dayof/<key>/ (baseline.json, enable_seedX.json, report.md).
#
# El domingo: correr una vez por layout revelado (el selector de nodos evita chocar
# con otros playbooks/jobs propios). Requiere: source scripts/env.sh hecho por dentro.

set -eo pipefail
cd "$(dirname "$0")/.."

ARG=${1:?uso: prepare_new_layout.sh <layout_file|builtin> [seeds] [steps]}
N_SEEDS=${2:-2}
STEPS=${3:-8000000}
SEED0=${SEED0:-400}

if [ -f "$ARG" ]; then
  # modo archivo .layout custom (from_grid)
  LAYOUT_FILE="$ARG"
  KEY=$(basename "$LAYOUT_FILE" .layout)
  LAYOUT_ARGS=(--layout-file "$LAYOUT_FILE")
  MF_FILE="$LAYOUT_FILE"
else
  # modo builtin por nombre (from_layout_name oficial); "-" en el manifest
  LAYOUT_FILE=""
  KEY="$ARG"
  LAYOUT_ARGS=(--layout "$KEY")
  MF_FILE="-"
fi
OUT="outputs/dayof/$KEY"
mkdir -p "$OUT" logs
REPORT="$OUT/report.md"
T_START=$(date +%s)

# --- entorno (modulos + venv; .venv necesita el modulo python3 por su libpython) ---
source scripts/env.sh >/dev/null

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/playbook.log"; }
mins() { echo $(( ($(date +%s) - $1) / 60 )); }

echo "# Playbook dia-de-competencia — $KEY" > "$REPORT"
echo "" >> "$REPORT"
echo "Inicio: $(date '+%Y-%m-%d %H:%M:%S') · seeds=$N_SEEDS · steps=$STEPS" >> "$REPORT"
log "=== $KEY: baseline del planner ==="

# ---------------------------------------------------------------- 1. baseline
T0=$(date +%s)
if ! python -m scripts.planner_baseline "${LAYOUT_ARGS[@]}" \
      --out "$OUT/baseline.json" 2>&1 | tee -a "$OUT/playbook.log"; then
  log "ABORTADO: el planner no produce sopas en $KEY. Arreglar planner/layout antes de entrenar."
  echo -e "\n**ABORTADO en baseline** (planner sin sopas — ver baseline.json)" >> "$REPORT"
  exit 1
fi
echo -e "\n## Baseline planner ($(mins $T0) min)\n\n\`\`\`json\n$(cat "$OUT/baseline.json")\n\`\`\`" >> "$REPORT"

# ---------------------------------------------------------------- 2. datos BC
log "=== $KEY: dataset BC (planner vs mixtos) ==="
T0=$(date +%s)
python -m training.collect_bc_data "${LAYOUT_ARGS[@]}" --episodes 600 \
    --out "data/bc/$KEY.npz" 2>&1 | tail -3 | tee -a "$OUT/playbook.log"
echo -e "\n## Dataset BC: data/bc/$KEY.npz ($(mins $T0) min)" >> "$REPORT"

# ---------------------------------------------------------------- 3. manifest
MANIFEST="training/jobs_dayof_$KEY.txt"
: > "$MANIFEST"
for i in $(seq 0 $((N_SEEDS - 1))); do
  SEED=$((SEED0 + i))
  echo "$KEY | $MF_FILE | $SEED | $STEPS | --partner solo_heavy --anneal-frac 0.8 --bc-data data/bc/$KEY.npz --bc-epochs 8" >> "$MANIFEST"
done
log "manifest: $MANIFEST ($N_SEEDS seeds desde $SEED0)"

# ------------------------------------------------- 4. lanzar (1 job por nodo)
# Pool de nodos-host CPU (docs/CLUSTER_NOTES.md); los GPU sirven sin --gres.
POOL=(n003 n004 n005 n006 ag001 ds001 g002)
part_of() { case "$1" in n0*) echo standard;; *) echo gpu;; esac; }

free_node() {  # primer nodo del pool sin jobs mios (recalcula en cada llamada)
  local busy; busy=$(squeue -u "$USER" -h -o "%N" | tr ',' '\n' | sort -u)
  for n in "${POOL[@]}"; do
    if ! grep -qx "$n" <<< "$busy"; then echo "$n"; return 0; fi
  done
  return 1
}

acct_slot() {  # cuenta con slot libre: tesis (<3 running+pending) o pregrado (<2)
  local nt np
  nt=$(squeue -u "$USER" -h -o "%a" | grep -c tesis || true)
  np=$(squeue -u "$USER" -h -o "%a" | grep -c pregrado || true)
  if [ "$nt" -lt 3 ]; then echo "tesis a-tesis 06:00:00"; return 0; fi
  if [ "$np" -lt 2 ]; then echo "pregrado a-pregrado 07:00:00"; return 0; fi
  return 1
}

T_TRAIN=$(date +%s)
JIDS=()
for i in $(seq 0 $((N_SEEDS - 1))); do
  SEED=$((SEED0 + i))
  # seccion critica entre playbooks concurrentes: la consulta de nodos/cuentas y el
  # sbatch deben ser atomicos o dos procesos eligen el mismo slot a la vez
  exec 9>>"outputs/dayof/.sbatch.lock"
  flock 9
  if SLOT=$(acct_slot) && NODE=$(free_node); then
    read -r ACC QOS TLIM <<< "$SLOT"
    JID=$(sbatch --parsable --account="$ACC" --qos="$QOS" --time="$TLIM" \
          --partition="$(part_of "$NODE")" --nodelist="$NODE" --array="$i-$i" \
          --export=ALL,JOBS="$MANIFEST" sbatch/train/run_train_ppo.sh)
    log "seed $SEED -> job ${JID} en $NODE (cuenta $ACC)"
  else
    # Sin slot/nodo: encadenar tras mi job mas antiguo, heredando su nodo (seguro
    # frente al bug del epilog: el nodo queda libre justo cuando aquel termina).
    OLD=$(squeue -u "$USER" -h -o "%i %N" -S i | head -1)
    OJID=${OLD%% *}; ONODE=${OLD##* }
    JID=$(sbatch --parsable --account=tesis --qos=a-tesis --time=06:00:00 \
          --partition="$(part_of "$ONODE")" --nodelist="$ONODE" --array="$i-$i" \
          --dependency=afterany:"$OJID" --export=ALL,JOBS="$MANIFEST" \
          sbatch/train/run_train_ppo.sh)
    log "seed $SEED -> job ${JID} ENCOLADO tras $OJID en $ONODE (dependency)"
  fi
  flock -u 9
  exec 9>&-
  JIDS+=("$JID")
  sleep 2
done
echo -e "\n## Entrenamientos lanzados: jobs ${JIDS[*]}" >> "$REPORT"

# ------------------------------------------------------------- 5. monitorear
log "esperando jobs: ${JIDS[*]} (poll cada 180s)"
ALL=$(IFS=,; echo "${JIDS[*]}")
for _ in $(seq 1 200); do
  ST=$(squeue -j "$ALL" -h -o "%T" 2>/dev/null | sort -u | tr '\n' ' ')
  [ -z "$ST" ] && break
  log "  estado: ${ST:-terminados} ($(mins $T_TRAIN) min de entrenamiento)"
  sleep 180
done
echo -e "\n## Entrenamiento: $(mins $T_TRAIN) min de pared" >> "$REPORT"

# ----------------------------------------------- 6. enable-check (con sonda)
log "=== $KEY: enable-check robusto por seed ==="
T0=$(date +%s)
ENABLED=""
echo -e "\n## Enable-check (PPO vs planner, 3 companeros)\n" >> "$REPORT"
for i in $(seq 0 $((N_SEEDS - 1))); do
  SEED=$((SEED0 + i))
  MODEL="models/$KEY/seed$SEED/best.zip"
  if [ ! -f "$MODEL" ]; then
    log "seed $SEED: sin best.zip (job fallo?) — se omite"
    echo "- seed $SEED: sin best.zip" >> "$REPORT"
    continue
  fi
  python -m scripts.enable_model --layout "$KEY" ${LAYOUT_FILE:+--layout-file "$LAYOUT_FILE"} \
      --model "$MODEL" 2>&1 | tee "$OUT/enable_seed$SEED.json" | tail -3
  if grep -q '"robust": true' "$OUT/enable_seed$SEED.json"; then
    ENABLED="seed$SEED"
    echo "- **seed $SEED: ROBUSTO → HABILITADO** ✅" >> "$REPORT"
    log "seed $SEED ROBUSTO -> habilitado (best.zip canonico + enabled + terrain.key)"
    break     # el primero robusto gana; los demas quedan como respaldo
  else
    echo "- seed $SEED: no robusto (ver enable_seed$SEED.json)" >> "$REPORT"
    log "seed $SEED no robusto"
  fi
done
echo -e "\nEnable-check: $(mins $T0) min" >> "$REPORT"

# ------------------------------------------------------------------ 7. cierre
TOTAL=$(mins $T_START)
if [ -n "$ENABLED" ]; then
  log "=== $KEY LISTO: PPO $ENABLED HABILITADO (total ${TOTAL} min) ==="
  echo -e "\n# ✅ RESULTADO: PPO $ENABLED habilitado — total ${TOTAL} min" >> "$REPORT"
else
  log "=== $KEY: ningun seed robusto; la entrega queda en el planner (segura). total ${TOTAL} min ==="
  echo -e "\n# ⚠️ RESULTADO: sin PPO robusto — selector queda en planner (total ${TOTAL} min)" >> "$REPORT"
fi
log "reporte: $REPORT"
