#!/usr/bin/env bash
# Download the VideoCoS demo bundle into this repository and check it:
#   bench/data/...            inputs of 2 samples per task family (T5 Kubric-4D, T6 Ego2Act, T7 Physics-IQ Verified)
#   demo/examples/...         codified scenes of 8 inputs (45 tasks) for the reasoning stage
#   demo/reference_outputs/   what the paper's runs produced for the same samples
# then fetch the two demo room videos (T1-T4) from VSI-Bench, which the bundle does not redistribute.
#   bash scripts/download_demo.sh            # bundle + room videos
#   bash scripts/download_demo.sh --no-rooms # bundle only (the rooms are needed only to re-run codification on them)
set -euo pipefail
cd "$(dirname "$0")/.."
# The bundle: https://huggingface.co/datasets/Shoubin/VideoCoS-demo (VIDEOCOS_DEMO_URL overrides the URL; set
# HF_ENDPOINT for a Hugging Face mirror)
DEMO_URL=${VIDEOCOS_DEMO_URL:-${HF_ENDPOINT:-https://huggingface.co}/datasets/Shoubin/VideoCoS-demo/resolve/main/VideoCoS-demo.zip}
if [ ! -f VideoCoS-demo.zip ]; then
  echo "downloading $DEMO_URL"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o VideoCoS-demo.zip.part "$DEMO_URL"
  mv VideoCoS-demo.zip.part VideoCoS-demo.zip
fi
unzip -o -q VideoCoS-demo.zip
sha256sum -c --quiet demo/SHA256SUMS && echo "demo bundle: $(wc -l < demo/SHA256SUMS) files verified"
if [ "${1:-}" != "--no-rooms" ]; then
  python3 scripts/fetch_vsi_videos.py --demo
fi
