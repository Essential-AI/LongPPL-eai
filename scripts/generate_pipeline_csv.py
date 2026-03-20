"""
Generate a nox CSV to score all parquet partitions for a given data source.

For large partitions (programming, eai-crawl-journals), splits into row-range
chunks so each job processes a manageable number of docs (~500 docs/job).

Usage:
    # Generate CSV for one source
    python scripts/generate_pipeline_csv.py --source arxiv --output /tmp/pipeline_arxiv.csv

    # Generate CSV for all sources
    python scripts/generate_pipeline_csv.py --all --output /tmp/pipeline_all.csv

    # Launch
    ~/modmax/nox /tmp/pipeline_arxiv.csv
"""

import argparse
import sys


SOURCES = {
    "arxiv": {
        "gcs_data": "gs://consus-dataproc/ocr/arxiv/text/tokenized_fulldocs",
        "n_partitions": 138,
        "approx_docs_per_partition": 500,
    },
    "programming": {
        "gcs_data": "gs://consus-dataproc/ocr/programming/text/tokenized_fulldocs_0.424B",
        "n_partitions": 1,
        "approx_docs_per_partition": 5000,  # single file, needs row splitting
    },
    "eai-crawl-journals": {
        "gcs_data": "gs://consus-dataproc/ocr/eai-crawl-journals/text/tokenized_fulldocs_0.525B",
        "n_partitions": 1,
        "approx_docs_per_partition": 5000,
    },
    "science-and-math": {
        "gcs_data": "gs://consus-dataproc/ocr/science-and-math/text/tokenized_fulldocs_1.190B",
        "n_partitions": 4,
        "approx_docs_per_partition": 2000,
    },
    "ia-ascm": {
        "gcs_data": "gs://consus-dataproc/ocr/ia-ascm/text/tokenized_fulldocs",
        "n_partitions": 273,
        "approx_docs_per_partition": 500,
    },
    "library": {
        "gcs_data": "gs://consus-dataproc/ocr/library/text/tokenized_fulldocs_8.589B",
        "n_partitions": 31,
        "approx_docs_per_partition": 2000,
    },
}

# Target: ~500 docs per job for ~2-3 hour runtime on 1 GPU at 32K context
DOCS_PER_JOB = 500

NOX_TEMPLATE_REF = "nox_templates/longppl-score-partition.yml"
SCRIPT_GCS = "gs://consus-dataproc/ocr/ia-ascm/scripts/score_partition.py"
MODEL = "meta-llama/Meta-Llama-3.1-8B"


def generate_rows(source_name, info, job_prefix="lps"):
    """Generate CSV rows for a source. Returns list of dicts."""
    rows = []
    gcs_data = info["gcs_data"]
    n_parts = info["n_partitions"]
    docs_per_part = info["approx_docs_per_partition"]

    scores_base = gcs_data.rsplit("/text/", 1)[0] + "/scores/longppl"

    if n_parts == 1 and docs_per_part > DOCS_PER_JOB:
        # Single large file: split by row ranges
        n_chunks = max(1, docs_per_part // DOCS_PER_JOB)
        for chunk_idx in range(n_chunks):
            row_start = chunk_idx * DOCS_PER_JOB
            row_end = row_start + DOCS_PER_JOB
            if chunk_idx == n_chunks - 1:
                row_end = ""  # last chunk takes all remaining

            workload = f"{job_prefix}-{source_name[:8]}-r{row_start}"
            # For single-file sources, discover the actual filename at runtime
            # We use a placeholder; the nox template handles glob
            rows.append({
                "WORKLOAD": workload,
                "SOURCE": source_name,
                "INPUT_PATH": gcs_data,
                "OUTPUT_PATH": f"{scores_base}/rows_{row_start}.parquet",
                "ROW_START": str(row_start),
                "ROW_END": str(row_end),
            })
    else:
        # Multi-partition: one job per partition
        for part_idx in range(n_parts):
            workload = f"{job_prefix}-{source_name[:8]}-p{part_idx:04d}"
            rows.append({
                "WORKLOAD": workload,
                "SOURCE": source_name,
                "INPUT_PATH": gcs_data,
                "OUTPUT_PATH": f"{scores_base}/part_{part_idx:04d}.parquet",
                "PARTITION_INDEX": str(part_idx),
                "ROW_START": "0",
                "ROW_END": "",
            })

            # If partition is large, split further
            if docs_per_part > DOCS_PER_JOB:
                n_chunks = docs_per_part // DOCS_PER_JOB
                rows.pop()  # remove the full-partition row
                for chunk_idx in range(n_chunks):
                    row_start = chunk_idx * DOCS_PER_JOB
                    row_end = row_start + DOCS_PER_JOB
                    if chunk_idx == n_chunks - 1:
                        row_end = ""
                    workload = f"{job_prefix}-{source_name[:8]}-p{part_idx:04d}-r{row_start}"
                    rows.append({
                        "WORKLOAD": workload,
                        "SOURCE": source_name,
                        "INPUT_PATH": gcs_data,
                        "OUTPUT_PATH": f"{scores_base}/part_{part_idx:04d}_rows_{row_start}.parquet",
                        "PARTITION_INDEX": str(part_idx),
                        "ROW_START": str(row_start),
                        "ROW_END": str(row_end),
                    })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate nox CSV for LongPPL scoring pipeline")
    parser.add_argument("--source", choices=list(SOURCES.keys()), help="Single source to generate")
    parser.add_argument("--all", action="store_true", help="Generate for all sources")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limit total jobs (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    if not args.source and not args.all:
        print("ERROR: specify --source or --all")
        sys.exit(1)

    sources = list(SOURCES.keys()) if args.all else [args.source]
    all_rows = []
    for src in sources:
        rows = generate_rows(src, SOURCES[src])
        all_rows.extend(rows)
        print(f"  {src}: {len(rows)} jobs")

    if args.max_jobs:
        all_rows = all_rows[:args.max_jobs]
        print(f"  (limited to {args.max_jobs} jobs)")

    print(f"\nTotal: {len(all_rows)} jobs")

    # Estimate compute
    # ~5 min per doc at 32K on MI300X, ~500 docs/job = ~42 hours/job
    # But with sliding window, actual throughput is ~2 min/doc → ~17 hours/job
    est_hours_per_job = DOCS_PER_JOB * 2 / 60
    est_gpu_hours = len(all_rows) * est_hours_per_job
    print(f"Estimated: ~{est_hours_per_job:.0f} hours/job, ~{est_gpu_hours:.0f} GPU-hours total")

    if args.dry_run:
        print("\n[dry run, not writing]")
        for row in all_rows[:5]:
            print(f"  {row}")
        if len(all_rows) > 5:
            print(f"  ... ({len(all_rows) - 5} more)")
        return

    # Write CSV
    columns = ["WORKLOAD", "USERNAME", "SOURCE", "MODEL", "SCRIPT_GCS",
               "INPUT_PATH", "OUTPUT_PATH", "PARTITION_INDEX", "ROW_START", "ROW_END"]

    with open(args.output, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in all_rows:
            vals = [
                row["WORKLOAD"],
                "kurt",
                row["SOURCE"],
                MODEL,
                SCRIPT_GCS,
                row["INPUT_PATH"],
                row["OUTPUT_PATH"],
                row.get("PARTITION_INDEX", "0"),
                row.get("ROW_START", "0"),
                row.get("ROW_END", ""),
            ]
            f.write(",".join(vals) + "\n")
        f.write(f"\n{NOX_TEMPLATE_REF},,,,,,,,,,\n")

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
