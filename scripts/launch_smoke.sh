#!/usr/bin/env bash
# Upload smoke_key_ratio.py to GCS and launch the K8s job via modmax.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Uploading smoke_key_ratio.py to GCS..."
gsutil cp "$SCRIPT_DIR/smoke_key_ratio.py" gs://consus-dataproc/ocr/ia-ascm/scripts/smoke_key_ratio.py

echo "Launching K8s job..."
cd ~/modmax && python launch_inference_server.py "$SCRIPT_DIR/smoke_key_ratio_k8s.yaml"

echo ""
echo "Watch logs with:"
echo "  kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=smoke-key-ratio"
echo ""
echo "Check results with:"
echo "  gsutil cat gs://consus-dataproc/ocr/ia-ascm/smoke_results.json"
