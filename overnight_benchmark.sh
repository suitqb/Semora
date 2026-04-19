#!/usr/bin/env bash
# =============================================================================
# overnight_benchmark.sh — Run complet nuit pour résultats analysables
#
# Usage:
#   ./overnight_benchmark.sh [MODELES]
#   ./overnight_benchmark.sh                          # utilise les modèles enabled dans models.yaml
#   ./overnight_benchmark.sh mistral-medium,molmo2-8b # modèles spécifiques
#
# Structure des runs produits :
#   extraction  × {no-tracker, tracker} × N_RUNS répétitions
#   complexity  × {no-tracker}          × 1  répétition
#
# Logs : overnight_benchmark.log (tout), overnight_benchmark_summary.log (résumé)
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python3"
RUNNER="run_benchmark.py"
MODELS_CFG="configs/models.yaml"
BENCHMARK_CFG="configs/benchmark.yaml"
CLIPS_EXTRACTION="configs/clips_extraction.yaml"
CLIPS_COMPLEXITY="configs/clips_complexity.yaml"

N_RUNS=2                     # répétitions par configuration extraction
LOG="overnight_benchmark.log"
SUMMARY="overnight_benchmark_summary.log"

# Modèles : argument CLI ou tous les enabled dans models.yaml
if [[ "${1:-}" != "" ]]; then
    MODELS_ARG="--models $1"
else
    MODELS_ARG="--models mistral-medium,mistral-large,gpt-4o-mini,gpt-5-mini,molmo2-8b"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
sep() { echo "$(printf '─%.0s' {1..78})" | tee -a "$LOG"; }

run_ok=0
run_fail=0
declare -a summary_lines=()

run_bench() {
    local label="$1"
    local mode="$2"
    local tracking_flag="$3"   # "true" | "false"
    local run_num="$4"

    local tag="${mode}__tracker-${tracking_flag}__run${run_num}"
    local tmp_cfg
    tmp_cfg=$(mktemp /tmp/benchmark_XXXX.yaml)

    # Injecter tracking dans une copie temporaire du benchmark.yaml
    python3 -c "
import yaml, sys
with open('$BENCHMARK_CFG') as f:
    cfg = yaml.safe_load(f)
cfg['benchmark']['features']['tracking'] = $([[ $tracking_flag == 'true' ]] && echo True || echo False)
cfg['benchmark']['run_id'] = None   # timestamp auto
with open('$tmp_cfg', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"

    log "▶  DÉBUT  [$tag]  — $label"
    sep

    local clips_cfg
    if [[ "$mode" == "complexity" ]]; then
        clips_cfg="$CLIPS_COMPLEXITY"
    else
        clips_cfg="$CLIPS_EXTRACTION"
    fi

    local start
    start=$(date +%s)

    set +e
    "$PYTHON" "$RUNNER" "$mode" \
        --non-interactive \
        $MODELS_ARG \
        --clips-cfg  "$clips_cfg" \
        --benchmark-cfg "$tmp_cfg" \
        2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    set -e

    rm -f "$tmp_cfg"

    local elapsed=$(( $(date +%s) - start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    if [[ $rc -eq 0 ]]; then
        log "✔  OK     [$tag]  (${mins}m${secs}s)"
        (( run_ok++ )) || true
        summary_lines+=("  ✔  $tag  (${mins}m${secs}s)")
    else
        log "✗  ERREUR [$tag]  code=$rc  (${mins}m${secs}s)"
        (( run_fail++ )) || true
        summary_lines+=("  ✗  $tag  code=$rc  (${mins}m${secs}s)")
    fi
    sep
}

# ── Début ─────────────────────────────────────────────────────────────────────

START_TOTAL=$(date +%s)
: > "$LOG"    # reset log

log "═══════════════════════════════════════════════════════════════════════════"
log "  overnight_benchmark.sh  —  démarrage $(date '+%Y-%m-%d %H:%M:%S')"
log "  modèles : ${1:-'enabled dans models.yaml'}"
log "  N_RUNS extraction : $N_RUNS"
log "═══════════════════════════════════════════════════════════════════════════"

# ── Plan 1 & 2 — Extraction  (sans tracker) ───────────────────────────────────
for i in $(seq 1 $N_RUNS); do
    run_bench "Extraction sans tracker  run $i/$N_RUNS" \
              "extraction" "false" "$i"
done

# ── Plan 1 & 2 — Extraction  (avec tracker) ───────────────────────────────────
for i in $(seq 1 $N_RUNS); do
    run_bench "Extraction avec tracker  run $i/$N_RUNS" \
              "extraction" "true" "$i"
done

# ── Plan 3 — Complexity  (sans tracker) ──────────────────────────────────────
for i in $(seq 1 $N_RUNS); do
    run_bench "Complexity sans tracker  run $i/$N_RUNS" \
              "complexity" "false" "$i"
done

# ── Plan 3 — Complexity  (avec tracker) ──────────────────────────────────────
for i in $(seq 1 $N_RUNS); do
    run_bench "Complexity avec tracker  run $i/$N_RUNS" \
              "complexity" "true" "$i"
done

# ── Résumé ────────────────────────────────────────────────────────────────────

ELAPSED_TOTAL=$(( $(date +%s) - START_TOTAL ))
HOURS=$(( ELAPSED_TOTAL / 3600 ))
MINS=$(( (ELAPSED_TOTAL % 3600) / 60 ))
SECS=$(( ELAPSED_TOTAL % 60 ))

{
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo "  RÉSUMÉ — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Durée totale : ${HOURS}h${MINS}m${SECS}s"
    echo "  Réussis : $run_ok   Échoués : $run_fail"
    echo "───────────────────────────────────────────────────────────────────────────"
    for line in "${summary_lines[@]}"; do
        echo "$line"
    done
    echo "═══════════════════════════════════════════════════════════════════════════"
} | tee -a "$LOG" | tee "$SUMMARY"

log "Runs dans : $SCRIPT_DIR/runs/"
log "Log complet    : $LOG"
log "Résumé         : $SUMMARY"

[[ $run_fail -eq 0 ]] && exit 0 || exit 1
