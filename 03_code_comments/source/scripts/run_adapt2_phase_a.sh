#!/usr/bin/env bash
# QECO-ADAPT2 Phase A launcher (see docs/qeco_adapt2_design.md).
#
# Batch 1 workers (fixed density, seed 42, 500 episodes, family adapt2):
#   A-1 sparse ablation @ users=100 : qeco, qeco-gate-only, qeco-adapt-general, qeco-adaptive-gate
#   A-0 identity check  @ users=300 : qeco-adapt2, qeco-adapt-general
#
# Results land in results/adapt2/fixed_density/<algorithm>/run_*.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT_DIR/scripts/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="$ROOT_DIR/env_channel/bin/python"
WORKFLOW="$ROOT_DIR/scripts/qeco_workflow.py"
LOG_DIR="$ROOT_DIR/logs/adapt2"
SEED=42
EPISODES=500

export CUDA_VISIBLE_DEVICES=-1
export QECO_AGENT_SESSION_MODE=per-agent
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export PYTHONUNBUFFERED=1

declare -a ALGORITHMS=(
  qeco
  qeco-gate-only
  qeco-adapt-general
  qeco-adaptive-gate
  qeco-adapt2
  qeco-adapt-general
)
declare -a USERS=(
  100
  100
  100
  100
  300
  300
)
declare -a SESSIONS=(
  adapt2_a1_u100_qeco
  adapt2_a1_u100_gate_only
  adapt2_a1_u100_general
  adapt2_a1_u100_adaptive_gate
  adapt2_a0_u300_adapt2
  adapt2_a0_u300_general
)
declare -a LOG_NAMES=(
  a1_user100_qeco_seed42.log
  a1_user100_qeco-gate-only_seed42.log
  a1_user100_qeco-adapt-general_seed42.log
  a1_user100_qeco-adaptive-gate_seed42.log
  a0_user300_qeco-adapt2_seed42.log
  a0_user300_qeco-adapt-general_seed42.log
)

usage() {
  cat <<'EOF'
Usage: scripts/run_adapt2_phase_a.sh {start|worker|status|compare}

  start                          Launch six adapt2 Phase A workers in tmux sessions.
  worker <algorithm> <users> <log> Run one algorithm inside its tmux session.
  status                         Show tmux sessions and active QECO workers.
  compare                        Create A-1 (users=100) and A-0 (users=300) comparisons.
EOF
}

run_worker() {
  local algorithm="$1"
  local users="$2"
  local log_name="$3"
  local status

  mkdir -p "$LOG_DIR"
  set +e
  "$PYTHON_BIN" "$WORKFLOW" \
    --family adapt2 \
    --users "$users" \
    --seed "$SEED" \
    run \
    --algorithms "$algorithm" \
    --episodes "$EPISODES" \
    2>&1 | tee "$LOG_DIR/$log_name"
  status=${PIPESTATUS[0]}
  set -e

  printf '\n[EXIT STATUS] %s\n' "$status" | tee -a "$LOG_DIR/$log_name"
  return "$status"
}

start_all() {
  local index algorithm users session log_name

  mkdir -p "$LOG_DIR"
  for index in "${!ALGORITHMS[@]}"; do
    algorithm="${ALGORITHMS[$index]}"
    users="${USERS[$index]}"
    session="${SESSIONS[$index]}"
    log_name="${LOG_NAMES[$index]}"

    if tmux has-session -t "$session" 2>/dev/null; then
      printf 'Existing tmux session: %s\n' "$session" >&2
      return 1
    fi

    : > "$LOG_DIR/$log_name"
    tmux new-session -d -s "$session" "$SELF worker $algorithm $users $log_name"
    printf 'Started %s (users=%s) in tmux session %s\n' "$algorithm" "$users" "$session"
  done
}

show_status() {
  tmux ls 2>/dev/null || true
  ps -u "$USER" -o pid,etime,%cpu,%mem,stat,args -ww \
    | grep '[c]hannel_common_eval.py' || true
  local run_root="$ROOT_DIR/results/adapt2/fixed_density"
  if [ -d "$run_root" ]; then
    find "$run_root" -name progress.json -exec sh -c \
      'printf "%s: " "$1"; cat "$1" | tr -d "\n"; printf "\n"' _ {} \; 2>/dev/null || true
  fi
}

run_comparison() {
  mkdir -p "$LOG_DIR"
  "$PYTHON_BIN" "$WORKFLOW" \
    --family adapt2 \
    --users 100 \
    --seed "$SEED" \
    compare \
    --algorithms qeco qeco-gate-only qeco-adapt-general qeco-adaptive-gate \
    2>&1 | tee "$LOG_DIR/a1_user100_compare_seed42.log"
  "$PYTHON_BIN" "$WORKFLOW" \
    --family adapt2 \
    --users 300 \
    --seed "$SEED" \
    compare \
    --algorithms qeco-adapt2 qeco-adapt-general \
    2>&1 | tee "$LOG_DIR/a0_user300_compare_seed42.log"
}

case "${1:-}" in
  start)
    start_all
    ;;
  worker)
    [ "$#" -eq 4 ] || { usage >&2; exit 2; }
    run_worker "$2" "$3" "$4"
    ;;
  status)
    show_status
    ;;
  compare)
    run_comparison
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
