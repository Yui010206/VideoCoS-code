#!/usr/bin/env bash
# Stop the server started by serve_h3.sh: its whole process group, by the PID it recorded.   bash stop_h3.sh [port]
PORT=${1:-30011}
ROOT=$(cd "$(dirname "$0")" && pwd)
PID=$(cat "$ROOT/logs/serve_$PORT.pid")
kill -TERM -- "-$PID" && echo "stopped sglang process group $PID (port $PORT)"
