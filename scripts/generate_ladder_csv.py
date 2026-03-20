"""
Generate a nox CSV for multi-context-length LongPPL scoring.

Generates 24 rows (4 pods/source x 6 sources), each processing 25 docs.

Usage:
    python scripts/generate_ladder_csv.py --output /tmp/longppl-context-ladder.csv
    python scripts/generate_ladder_csv.py --source arxiv --output /tmp/ladder_arxiv.csv
    python scripts/generate_ladder_csv.py --pods-per-source 2 --docs-per-pod 50 --output /tmp/ladder.csv

    # Launch
    ~/modmax/nox /tmp/longppl-context-ladder.csv
"""

import argparse
import sys


SOURCES = {
    "arxiv": "gs://consus-dataproc/ocr/arxiv/text/tokenized_fulldocs",
    "programming": "gs://consus-dataproc/ocr/programming/text/tokenized_fulldocs_0.424B",
    "eai-crawl-journals": "gs://consus-dataproc/ocr/eai-crawl-journals/text/tokenized_fulldocs_0.525B",
    "science-and-math": "gs://consus-dataproc/ocr/science-and-math/text/tokenized_fulldocs_1.190B",
    "ia-ascm": "gs://consus-dataproc/ocr/ia-ascm/text/tokenized_fulldocs",
    "library": "gs://consus-dataproc/ocr/library/text/tokenized_fulldocs_8.589B",
}

# Results base: alongside the source data
RESULTS_BASE = "gs://consus-dataproc/ocr"

NOX_TEMPLATE_REF = "nox_templates/longppl-context-ladder.yml"
SCRIPT_GCS = "gs://consus-dataproc/ocr/ia-ascm/scripts/context_ladder.py"
# min tokens = smallest context (4096) + score window (10240)
MIN_TOKENS = 14336


def generate_rows(sources, pods_per_source, docs_per_pod):
    """Generate CSV rows for context ladder jobs."""
    rows = []
    for source_name, gcs_data in sources.items():
        # Extract source path component for results
        # e.g., gs://consus-dataproc/ocr/arxiv/text/... -> arxiv
        source_key = source_name

        for pod_idx in range(pods_per_source):
            doc_offset = pod_idx * docs_per_pod
            workload = f"ladder-{source_key[:8]}-{pod_idx}"
            results_path = f"{RESULTS_BASE}/{source_key}/scores/context_ladder/ladder_{pod_idx}.json"

            rows.append({
                "WORKLOAD": workload,
                "GCS_DATA_PATH": gcs_data,
                "N_DOCS": str(docs_per_pod),
                "DOC_OFFSET": str(doc_offset),
                "MIN_TOKENS": str(MIN_TOKENS),
                "RESULTS_GCS_PATH": results_path,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate nox CSV for context ladder scoring"
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--source", choices=list(SOURCES.keys()),
        help="Single source (default: all sources)",
    )
    parser.add_argument(
        "--pods-per-source", type=int, default=4,
        help="Number of pods per source (default: 4)",
    )
    parser.add_argument(
        "--docs-per-pod", type=int, default=25,
        help="Number of docs per pod (default: 25)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    if args.source:
        sources = {args.source: SOURCES[args.source]}
    else:
        sources = SOURCES

    rows = generate_rows(sources, args.pods_per_source, args.docs_per_pod)
    total_docs = len(rows) * args.docs_per_pod

    print(f"Sources: {len(sources)}")
    print(f"Pods per source: {args.pods_per_source}")
    print(f"Docs per pod: {args.docs_per_pod}")
    print(f"Total jobs: {len(rows)}")
    print(f"Total docs: {total_docs}")

    if args.dry_run:
        print("\n[dry run]")
        for row in rows[:6]:
            print(f"  {row['WORKLOAD']}: offset={row['DOC_OFFSET']} n={row['N_DOCS']}")
        if len(rows) > 6:
            print(f"  ... ({len(rows) - 6} more)")
        return

    columns = [
        "WORKLOAD", "USERNAME", "GCS_DATA_PATH", "N_DOCS", "DOC_OFFSET",
        "MIN_TOKENS", "RESULTS_GCS_PATH",
    ]

    with open(args.output, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            vals = [
                row["WORKLOAD"],
                "kurt",
                row["GCS_DATA_PATH"],
                row["N_DOCS"],
                row["DOC_OFFSET"],
                row["MIN_TOKENS"],
                row["RESULTS_GCS_PATH"],
            ]
            f.write(",".join(vals) + "\n")
        f.write(f"\n{NOX_TEMPLATE_REF},,,,,,\n")

    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
