#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN="$ROOT_DIR/scripts/run_dynamic_campaign.py"

usage() {
  cat <<'EOF'
Usage: scripts/launch_dynamic_cpu.sh \
  --python PATH \
  --config PATH \
  --output-dir PATH \
  --session-name NAME \
  [--seeds 42 77 123] \
  [--budget-seconds 21600] \
  [--cell-timeout-seconds 7200] \
  [--min-free-gib 5]
EOF
}

PYTHON_BIN=""
CONFIG=""
OUTPUT_DIR=""
SESSION_NAME=""
BUDGET_SECONDS=21600
CELL_TIMEOUT_SECONDS=7200
MIN_FREE_GIB=5
SEEDS=(42 77 123)

absolute_path() {
  if [[ "$1" == /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$PWD" "$1"
  fi
}

is_non_negative_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

is_positive_number() {
  is_non_negative_number "$1" && [[ ! "$1" =~ ^0+([.]0+)?$ ]]
}

while (($#)); do
  case "$1" in
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --session-name)
      SESSION_NAME="${2:-}"
      shift 2
      ;;
    --seeds)
      shift
      SEEDS=()
      while (($#)) && [[ "$1" != --* ]]; do
        SEEDS+=("$1")
        shift
      done
      ;;
    --budget-seconds)
      BUDGET_SECONDS="${2:-}"
      shift 2
      ;;
    --cell-timeout-seconds)
      CELL_TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    --min-free-gib)
      MIN_FREE_GIB="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PYTHON_BIN" || -z "$CONFIG" || -z "$OUTPUT_DIR" || -z "$SESSION_NAME" ]]; then
  usage >&2
  exit 2
fi
if ! is_positive_number "$BUDGET_SECONDS" || ! is_positive_number "$CELL_TIMEOUT_SECONDS"; then
  printf 'Time budgets must be finite positive decimal numbers.\n' >&2
  exit 2
fi
if ! is_non_negative_number "$MIN_FREE_GIB"; then
  printf 'Disk floor must be a finite non-negative decimal number.\n' >&2
  exit 2
fi
if ((${#SEEDS[@]} == 0)); then
  printf 'At least one seed is required.\n' >&2
  exit 2
fi
if [[ ! "$SESSION_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Session name may contain only letters, digits, underscore, and hyphen.\n' >&2
  exit 2
fi

PYTHON_BIN="$(absolute_path "$PYTHON_BIN")"
CONFIG="$(absolute_path "$CONFIG")"
OUTPUT_DIR="$(absolute_path "$OUTPUT_DIR")"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  printf 'Config does not exist: %s\n' "$CONFIG" >&2
  exit 2
fi
if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  printf 'Refusing to interfere with existing tmux session: %s\n' "$SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES=-1
export QECO_AGENT_SESSION_MODE=per-agent
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

COMMAND=(
  "$PYTHON_BIN"
  "$CAMPAIGN"
  --python "$PYTHON_BIN"
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --seeds "${SEEDS[@]}"
  --budget-seconds "$BUDGET_SECONDS"
  --cell-timeout-seconds "$CELL_TIMEOUT_SECONDS"
  --min-free-gib "$MIN_FREE_GIB"
)

tmux new-session -d -c "$ROOT_DIR" -s "$SESSION_NAME" "${COMMAND[@]}"
printf 'Started dynamic CPU campaign in tmux session %s\n' "$SESSION_NAME"
printf 'Output directory: %s\n' "$OUTPUT_DIR"
