#!/usr/bin/env bash
# QECO-ADAPT2 Phase B launcher (see docs/qeco_adapt2_design.md).
#
# Final random-density experiment (family adapt2, seed 42, 500 episodes):
#   agent pool 1500, per-episode active users sampled from {100,300,500,800,1200,1500}
#   algorithms: qeco, qeco-adapt-general, qeco-adaptive-gate, qeco-adapt2
#
# Results land in results/adapt2/random_density/<algorithm>/run_*.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT_DIR/scripts/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="$ROOT_DIR/env_channel/bin/python"
WORKFLOW="$ROOT_DIR/scripts/qeco_workflow.py"
LOG_DIR="$ROOT_DIR/logs/adapt2"
SEED=42
EPISODES=500
POOL_USERS=1500
DENSITY_CHOICES="100,300,500,800,1200,1500"

export CUDA_VISIBLE_DEVICES=-1
export QECO_AGENT_SESSION_MODE=per-agent
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export PYTHONUNBUFFERED=1

declare -a ALGORITHMS=(
  qeco
  qeco-adapt-general
  qeco-adaptive-gate
  qeco-adapt2
)
declare -a SESSIONS=(
  adapt2_b_rand_qeco
  adapt2_b_rand_general
  adapt2_b_rand_adaptive_gate
  adapt2_b_rand_adapt2
)
declare -a LOG_NAMES=(
  b_rand_qeco_seed42.log
  b_rand_qeco-adapt-general_seed42.log
  b_rand_qeco-adaptive-gate_seed42.log
  b_rand_qeco-adapt2_seed42.log
)

usage() {
  cat <<'EOF'
Usage: scripts/run_adapt2_phase_b.sh {start|worker|status|compare}

  start                    Launch four random-density workers in tmux sessions.
  worker <algorithm> <log> Run one algorithm inside its tmux session.
  status                   Show tmux sessions and active QECO workers.
  compare                  Create the Phase B comparison after all runs finish.
EOF
}

run_worker() {
  local algorithm="$1"
  local log_name="$2"
  local status_code

  mkdir -p "$LOG_DIR"
  set +e
  "$PYTHON_BIN" "$WORKFLOW" \
    --family adapt2 \
    --density-mode random \
    --density-choices "$DENSITY_CHOICES" \
    --users "$POOL_USERS" \
    --seed "$SEED" \
    --allow-1200 \
    run \
    --algorithms "$algorithm" \
    --episodes "$EPISODES" \
    2>&1 | tee "$LOG_DIR/$log_name"
  status_code=${PIPESTATUS[0]}
  set -e

  printf '\n[EXIT STATUS] %s\n' "$status_code" | tee -a "$LOG_DIR/$log_name"
  return "$status_code"
}

start_all() {
  local index algorithm session log_name

  mkdir -p "$LOG_DIR"
  for index in "${!ALGORITHMS[@]}"; do
    algorithm="${ALGORITHMS[$index]}"
    session="${SESSIONS[$index]}"
    log_name="${LOG_NAMES[$index]}"

    if tmux has-session -t "$session" 2>/dev/null; then
      printf 'Existing tmux session: %s\n' "$session" >&2
      return 1
    fi

    : > "$LOG_DIR/$log_name"
    tmux new-session -d -s "$session" "$SELF worker $algorithm $log_name"
    printf 'Started %s (pool=%s, random density) in tmux session %s\n' \
      "$algorithm" "$POOL_USERS" "$session"
  done
}

show_status() {
  tmux ls 2>/dev/null || true
  ps -u "$USER" -o pid,etime,%cpu,%mem,stat,args -ww \
    | grep '[c]hannel_common_eval.py' || true
  local run_root="$ROOT_DIR/results/adapt2/random_density"
  if [ -d "$run_root" ]; then
    find "$run_root" -name progress.json -exec sh -c \
      'printf "%s: " "$1"; cat "$1" | tr -d "\n"; printf "\n"' _ {} \; 2>/dev/null || true
  fi
}

run_comparison() {
  mkdir -p "$LOG_DIR"
  "$PYTHON_BIN" "$WORKFLOW" \
    --family adapt2 \
    --density-mode random \
    --density-choices "$DENSITY_CHOICES" \
    --users "$POOL_USERS" \
    --seed "$SEED" \
    --allow-1200 \
    compare \
    --algorithms qeco qeco-adapt-general qeco-adaptive-gate qeco-adapt2 \
    2>&1 | tee "$LOG_DIR/b_rand_compare_seed42.log"
}

case "${1:-}" in
  start)
    start_all
    ;;
  worker)
    [ "$#" -eq 3 ] || { usage >&2; exit 2; }
    run_worker "$2" "$3"
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
