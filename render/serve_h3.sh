#!/usr/bin/env bash
# Serve MiniMax-H3 Ref2VA with SGLang on the given GPUs:   bash serve_h3.sh <gpu[,gpu...]> [port]
# Log logs/serve_<port>.log, PID logs/serve_<port>.pid (stop with stop_h3.sh <port>). Ready when the log says
# "fired up"; if it also says "request broker exited", every request gets 503: stop it and use another port.
# Ports: besides PORT the server binds PORT+1 (its offline request broker) and a dist port, 30005 or, when that is
# taken, the next free one of 30042, 30079, 30116, ... (30005 + 37k). Choose PORT so that neither PORT nor PORT+1 is on
# that sequence or used by another server (30021, 30031, 30061, 30091 and 30013 work), and start servers one at a time.
# Topology by number of GPUs (H3_TOPOLOGY overrides it; 3 GPUs have no valid split of H3's 56 heads):
#   1 GPU : 30 of the 50 DiT layers resident, the rest and the text encoder offloaded layer-wise to pinned host memory
#   2 GPUs: TP2, text encoder offloaded layer-wise (the 1- and 2-GPU recipes are the ones the paper's runs used)
#   4 GPUs: TP2 x Ulysses2, the SGLang cookbook's fastest lossless recipe; 8 GPUs: TP2 x Ulysses4
# The full BF16/FP32 model, no quantization. Weights: MiniMaxAI/MiniMax-H3 (model_index.json + Ref2VA/*) at
# H3_MODEL_PATH; HF_HUB_OFFLINE stops the server from fetching anything else (the full repo is far larger).
set -euo pipefail
GPUS=${1:?usage: serve_h3.sh <gpu[,gpu...]> [port]}
PORT=${2:-30011}
MODEL=${H3_MODEL_PATH:?set H3_MODEL_PATH to the MiniMax-H3 folder (model_index.json + Ref2VA/)}
N=$(awk -F, '{print NF}' <<< "$GPUS")
case $N in
  1) TOPO="--dit-layerwise-offload true --dit-layerwise-resident-layers 30 --dit-layerwise-residency-lifetime permanent"
     TOPO="$TOPO --layerwise-offload-components text_encoder --pin-cpu-memory true" ;;
  2) TOPO="--tp-size 2 --layerwise-offload-components text_encoder" ;;
  4) TOPO="--tp-size 2 --ulysses-degree 2" ;;
  8) TOPO="--tp-size 2 --ulysses-degree 4" ;;
  *) TOPO="--ulysses-degree $N" ;;
esac
TOPO=${H3_TOPOLOGY:-$TOPO}
ROOT=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT/logs"
if [ ! -f "$MODEL/model_index.json" ] || [ ! -d "$MODEL/Ref2VA" ]; then
  echo "no MiniMax-H3 Ref2VA weights at $MODEL (set H3_MODEL_PATH)"; exit 1
fi
source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
conda activate h3render
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
# Started from Python with a new session, so the recorded PID is the server itself and its process-group id
# (`setsid cmd &` forks once more when the shell made the job a group leader, and $! is then a dead parent).
python - "$ROOT/logs/serve_$PORT.log" "$ROOT/logs/serve_$PORT.pid" \
  sglang serve --model-path "$MODEL" --model-variant ref2va --num-gpus "$N" $TOPO \
  --encoder-parallel auto --performance-mode speed --host 127.0.0.1 --port "$PORT" <<'EOF'
import subprocess, sys
log, pidfile, *command = sys.argv[1:]
process = subprocess.Popen(command, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           start_new_session=True, cwd=log.rsplit("/logs/", 1)[0])   # its outputs/ copies stay here
open(pidfile, "w").write(str(process.pid))
EOF
echo "sglang pid $(cat "$ROOT/logs/serve_$PORT.pid") on GPUs $GPUS ($TOPO), port $PORT, log $ROOT/logs/serve_$PORT.log"
