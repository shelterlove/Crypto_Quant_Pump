#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-cq}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
CONFIG_PATH="${CONFIG_PATH:-configs/main.yaml}"
STATE_PATH="${STATE_PATH:-paper_state/main.json}"
LOCK_PATH="${LOCK_PATH:-paper_state/paper.lock}"
LIVE_PATH="${LIVE_PATH:-paper_state/live_status.json}"
REPORT_DIR="${REPORT_DIR:-reports}"
MONITOR_PORT="${MONITOR_PORT:-8000}"
CYCLE_SLEEP_SECONDS="${CYCLE_SLEEP_SECONDS:-300}"
LIVE_SLEEP_SECONDS="${LIVE_SLEEP_SECONDS:-60}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p logs paper_state

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n cycle
tmux send-keys -t "$SESSION_NAME:cycle" \
  "cd '$PROJECT_DIR' && while true; do date -Is | tee -a logs/paper.log; uv run crypto-quant paper cycle --config '$CONFIG_PATH' --state-path '$STATE_PATH' --lock-path '$LOCK_PATH' --report-dir '$REPORT_DIR' --json-output | tee -a logs/paper.log; sleep '$CYCLE_SLEEP_SECONDS'; done" C-m

tmux new-window -t "$SESSION_NAME" -n live
tmux send-keys -t "$SESSION_NAME:live" \
  "cd '$PROJECT_DIR' && while true; do date -Is | tee -a logs/live.log; uv run crypto-quant paper refresh-live --config '$CONFIG_PATH' --state-path '$STATE_PATH' --out-path '$LIVE_PATH' --report-dir '$REPORT_DIR' | tee -a logs/live.log; sleep '$LIVE_SLEEP_SECONDS'; done" C-m

tmux new-window -t "$SESSION_NAME" -n monitor
tmux send-keys -t "$SESSION_NAME:monitor" \
  "cd '$PROJECT_DIR' && uv run crypto-quant paper serve-monitor --config '$CONFIG_PATH' --state-dir 'paper_state' --report-dir '$REPORT_DIR' --host 0.0.0.0 --port '$MONITOR_PORT'" C-m

echo "started tmux session: $SESSION_NAME"
echo "attach with: tmux attach -t $SESSION_NAME"
echo "monitor url: http://<server-ip>:$MONITOR_PORT/paper_state/dashboard.html"
