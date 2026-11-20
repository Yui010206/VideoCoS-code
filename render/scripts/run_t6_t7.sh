#!/usr/bin/env bash
# Render T7 and then T6 with the full MiniMax-H3 Ref2VA model on one or more SGLang servers (the final setup).
#
#   bash scripts/run_t6_t7.sh [--baseline] GPUS:PORT [GPUS:PORT ...]
#   bash scripts/run_t6_t7.sh 0,1:30021 2,3:30031          # two servers of two GPUs each
#
# Servers start one after another (two starting at once race for SGLang's dist port). serve_h3.sh picks the topology
# from the number of GPUs (1, 2 or 4). The next server starts only after this one's log says "fired up"; a server whose
# log also says "request broker exited" answers every request with 503, so it is stopped and skipped. As soon as a
# server is up its generator chain starts; all chains share each case list through claim files:
#   T7 ours     t6_h3/generate_t7_ref2va.py                                 -> runs/ours_h3full_t7  (63 pairs with a plan)
#   T6 ours     t6_h3/generate.py --name ours_h3_v4 --prompts prompts_v4/T6 -> runs/ours_h3_v4      (101 cases with a plan)
# and with --baseline, after those, the H3-only baselines (no plan video):
#   T7 H3 only  t6_h3/generate_t7_ref2va.py --no-reference                  -> runs/h3only_t7      (all 66 pairs)
#   T6 H3 only  t6_h3/generate_h3only_t6.py                                 -> runs/h3only_t6      (all 110 cases)
# When every chain has ended the servers are stopped. Ctrl-C or `kill <pid of this script>` stops the chains and the
# servers; running the same command again resumes (finished cases are skipped, claims of dead generators are taken over).
# Logs: logs/run_t6_t7.log (this script), logs/serve_<port>.log (servers), logs/chain_<port>.log (generators).
# Needs the prepared inputs (README, step 2). Env overrides: H3_PYTHON (python of the h3render env), H3_MODEL_PATH,
# CONDA_SH (passed on to serve_h3.sh).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${H3_PYTHON:-python}
export H3_MODEL_PATH=${H3_MODEL_PATH:?set H3_MODEL_PATH to the MiniMax-H3 folder (model_index.json + Ref2VA/)}
BASELINE=0
if [ "${1:-}" = "--baseline" ]; then BASELINE=1; shift; fi
if [ $# -lt 1 ]; then sed -n '4,5p' "$0"; exit 1; fi
mkdir -p "$ROOT/logs"
LOG=$ROOT/logs/run_t6_t7.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# --- inputs (cheap CPU steps, run them first; see README)
n6=$(ls "$ROOT"/inputs/T6/*/case.json 2>/dev/null | wc -l)
np=$(ls "$ROOT"/prompts_v4/T6/*.txt 2>/dev/null | wc -l)
n7=$(ls "$ROOT"/inputs/T7/*/last_observed.png 2>/dev/null | wc -l)
if [ "$n6" -eq 0 ] || [ "$np" -eq 0 ] || [ "$n7" -eq 0 ]; then
  echo "inputs missing (inputs/T6: $n6 cases, prompts_v4/T6: $np prompts, inputs/T7: $n7 keyframes); first run, in $ROOT:"
  echo "  $PY t6_h3/prepare.py && $PY t6_h3/make_prompts.py && $PY t6_h3/make_t7_inputs.py"
  exit 1
fi

# --- ports: PORT and PORT+1 (the server's request broker) must stay off SGLang's dist-port sequence 30005 + 37k
# (30005, 30042, 30079, 30116, ...), off each other and off anything already listening
on_dist_sequence() { [ "$1" -ge 30005 ] && [ $(( ($1 - 30005) % 37 )) -eq 0 ]; }
declare -A TAKEN=()
for spec in "$@"; do
  GPUS=${spec%%:*}; PORT=${spec##*:}
  if [[ "$spec" != *:* || ! "$PORT" =~ ^[0-9]+$ || ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "bad server spec '$spec' (want GPUS:PORT, e.g. 0,1:30021)"; exit 1
  fi
  for p in "$PORT" $((PORT + 1)); do
    if on_dist_sequence "$p"; then echo "server :$PORT: port $p is on SGLang's dist-port sequence 30005 + 37k"; exit 1; fi
    if [ -n "${TAKEN[$p]:-}" ]; then echo "server :$PORT: port $p is also needed by another server"; exit 1; fi
    if [ -n "$(ss -Hltn "sport = :$p" 2>/dev/null)" ]; then echo "server :$PORT: port $p is already in use"; exit 1; fi
    TAKEN[$p]=1
  done
done

gpus_free() {                            # every GPU in $1 has < 1000 MiB in use (nobody else's job is on it)
  local g m
  for g in ${1//,/ }; do
    m=$(timeout 30 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
    [ "${m:-99999}" -lt 1000 ] || return 1
  done
}

STARTED=()                               # ports of the servers this script started
CHAINS=()                                # PIDs of the generator chains (one subshell per server)
stop_servers() { local p; for p in "${STARTED[@]}"; do bash "$ROOT/stop_h3.sh" "$p" >> "$LOG" 2>&1; done; }
on_signal() {
  trap - INT TERM
  say "stopping: generator chains ${CHAINS[*]:-none}, servers ${STARTED[*]:-none}"
  local c
  for c in "${CHAINS[@]}"; do            # freeze the chain so it cannot start its next step, end its generator, drop it
    kill -STOP "$c" 2>/dev/null; pkill -TERM -P "$c" 2>/dev/null; kill -KILL "$c" 2>/dev/null
  done
  stop_servers
  exit 130
}
trap on_signal INT TERM

say "run_t6_t7 started (pid $$): servers $*, baselines $([ $BASELINE -eq 1 ] && echo yes || echo no)"
for spec in "$@"; do
  GPUS=${spec%%:*}; PORT=${spec##*:}; URL=http://127.0.0.1:$PORT
  if ! gpus_free "$GPUS"; then say "GPUs $GPUS are in use; server :$PORT skipped"; continue; fi
  if ! bash "$ROOT/serve_h3.sh" "$GPUS" "$PORT" >> "$LOG" 2>&1; then say "serve_h3.sh failed for :$PORT (see $LOG)"; continue; fi
  STARTED+=("$PORT")
  up=0
  for _ in $(seq 1 360); do              # up to 3 h: a cold read of the weights over the NAS is slow
    grep -q "fired up" "$ROOT/logs/serve_$PORT.log" 2>/dev/null && { up=1; break; }
    kill -0 "$(cat "$ROOT/logs/serve_$PORT.pid")" 2>/dev/null || break
    sleep 30
  done
  if [ $up -eq 0 ]; then
    say "server :$PORT on GPUs $GPUS did not come up; see logs/serve_$PORT.log"
    bash "$ROOT/stop_h3.sh" "$PORT" >> "$LOG" 2>&1; continue
  fi
  if grep -q "request broker exited" "$ROOT/logs/serve_$PORT.log"; then
    say "server :$PORT came up with a dead request broker (port clash, every request would get 503); stopped it"
    bash "$ROOT/stop_h3.sh" "$PORT" >> "$LOG" 2>&1; continue
  fi
  say "server :$PORT up on GPUs $GPUS"
  (
    cd "$ROOT" || exit 1
    "$PY" -u t6_h3/generate_t7_ref2va.py --server "$URL" --ids all
    "$PY" -u t6_h3/generate.py --server "$URL" --ids all --name ours_h3_v4 --prompts "$ROOT/prompts_v4/T6" --wait-minutes 30
    if [ "$BASELINE" -eq 1 ]; then
      "$PY" -u t6_h3/generate_t7_ref2va.py --server "$URL" --ids all --no-reference
      "$PY" -u t6_h3/generate_h3only_t6.py --server "$URL" --ids all --wait-minutes 30
    fi
  ) > "$ROOT/logs/chain_$PORT.log" 2>&1 < /dev/null &
  CHAINS+=("$!")
  say "generator chain for :$PORT started (pid $!), log logs/chain_$PORT.log"
done

if [ ${#CHAINS[@]} -eq 0 ]; then say "no server came up; nothing rendered"; stop_servers; exit 1; fi
for c in "${CHAINS[@]}"; do wait "$c"; done
say "all generator chains ended; stopping the servers"
stop_servers
for run in ours_h3full_t7/T7 ours_h3_v4/T6 h3only_t7/T7 h3only_t6/T6; do
  [ -d "$ROOT/runs/$run" ] && say "runs/$run: $(ls "$ROOT"/runs/$run/*/video.mp4 2>/dev/null | wc -l) videos"
done
say "cases without video.mp4 (a failed request, a crashed server, a killed generator) are rendered by running the same" \
    "command again, or one generator with --ids <case>"
