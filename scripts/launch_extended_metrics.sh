#!/usr/bin/env bash
# Upload updated smoke_key_ratio.py (with extended metrics) and launch jobs.
#
# Usage:
#   Phase 2 (validation, 5 docs/source):
#     bash scripts/launch_extended_metrics.sh validate
#
#   Phase 3 (full, 100 docs/source):
#     bash scripts/launch_extended_metrics.sh full
#
#   Download results:
#     bash scripts/launch_extended_metrics.sh download [validate|full]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GCS_SCRIPT="gs://consus-dataproc/ocr/ia-ascm/scripts/smoke_key_ratio.py"
NOX="$HOME/modmax/nox"
NOX_DIR="$HOME/modmax/nox_templates/examples"

case "${1:-validate}" in
  validate)
    echo "=== Phase 2: Uploading script and launching validation (5 docs/source) ==="
    gsutil cp "$SCRIPT_DIR/smoke_key_ratio.py" "$GCS_SCRIPT"
    echo "Script uploaded to $GCS_SCRIPT"
    "$NOX" "$NOX_DIR/longppl-extended-validation.csv"
    echo ""
    echo "Watch logs:"
    echo "  kubectl logs -f -l essential.ai/username=kurt -l xpk.google.com/workload=longppl-ext-val-arxiv"
    echo ""
    echo "When done, download results with:"
    echo "  bash scripts/launch_extended_metrics.sh download validate"
    ;;

  full)
    echo "=== Phase 3: Uploading script and launching full run (100 docs/source) ==="
    gsutil cp "$SCRIPT_DIR/smoke_key_ratio.py" "$GCS_SCRIPT"
    echo "Script uploaded to $GCS_SCRIPT"
    "$NOX" "$NOX_DIR/longppl-extended-100.csv"
    echo ""
    echo "Watch logs:"
    echo "  kubectl logs -f -l essential.ai/username=kurt -l xpk.google.com/workload=longppl-ext-100-arxiv"
    echo ""
    echo "When done, download results with:"
    echo "  bash scripts/launch_extended_metrics.sh download full"
    ;;

  download)
    PHASE="${2:-validate}"
    SOURCES="arxiv programming eai-crawl-journals science-and-math ia-ascm library"
    if [ "$PHASE" = "validate" ]; then
      SUFFIX="extended_val"
    else
      SUFFIX="extended_100"
    fi
    OUTDIR="/tmp/longppl_${SUFFIX}"
    mkdir -p "$OUTDIR"
    echo "Downloading $PHASE results to $OUTDIR/"
    for src in $SOURCES; do
      echo "  $src..."
      gsutil cp "gs://consus-dataproc/ocr/$src/longppl_${SUFFIX}.json" "$OUTDIR/${src}.json" 2>/dev/null || echo "    (not ready yet)"
    done
    echo ""
    echo "Results in $OUTDIR/:"
    ls -la "$OUTDIR/" 2>/dev/null
    ;;

  *)
    echo "Usage: $0 {validate|full|download [validate|full]}"
    exit 1
    ;;
esac
