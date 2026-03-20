"""
Merge scored parquet partitions and produce filtering-ready output.

Reads all scored parquet files from a source's scores/longppl/ directory,
concatenates them, and writes:
  1. A single merged parquet with all docs + scores
  2. A filtered parquet with docs above threshold
  3. A summary JSON with distribution stats

Usage:
    python scripts/merge_scores.py \
        --source arxiv \
        --scores-path gs://consus-dataproc/ocr/arxiv/scores/longppl \
        --output-dir gs://consus-dataproc/ocr/arxiv/scores

    # With custom filter
    python scripts/merge_scores.py \
        --source arxiv \
        --scores-path gs://consus-dataproc/ocr/arxiv/scores/longppl \
        --output-dir gs://consus-dataproc/ocr/arxiv/scores \
        --filter-metric longppl_mean_context_benefit \
        --filter-percentile 25
"""

import argparse
import json
import sys

import numpy as np


SCORE_COLUMNS = [
    "longppl_key_token_ratio",
    "longppl_n_key",
    "longppl_n_total",
    "longppl_mean_context_benefit",
    "longppl_median_context_benefit",
    "longppl_context_benefit_std",
    "longppl_frac_positive_benefit",
    "longppl_weighted_context_benefit",
    "longppl_context_benefit_near",
    "longppl_context_benefit_mid",
    "longppl_context_benefit_far",
]


def main():
    parser = argparse.ArgumentParser(description="Merge scored partitions")
    parser.add_argument("--source", required=True)
    parser.add_argument("--scores-path", required=True, help="GCS path to scored parquet dir")
    parser.add_argument("--output-dir", required=True, help="GCS path for merged output")
    parser.add_argument("--filter-metric", default="longppl_mean_context_benefit",
                        help="Metric to filter on")
    parser.add_argument("--filter-percentile", type=int, default=25,
                        help="Keep docs above this percentile (0=keep all)")
    args = parser.parse_args()

    import gcsfs
    import pyarrow as pa
    import pyarrow.parquet as pq

    fs = gcsfs.GCSFileSystem()

    # Discover scored partitions
    scored_files = sorted(fs.glob(args.scores_path.replace("gs://", "") + "/*.parquet"))
    if not scored_files:
        print(f"ERROR: No scored parquet files found in {args.scores_path}")
        sys.exit(1)
    print(f"Found {len(scored_files)} scored partition(s)")

    # Read and concatenate
    tables = []
    for f_path in scored_files:
        with fs.open(f_path, "rb") as f:
            t = pq.read_table(f)
            tables.append(t)
        print(f"  {f_path}: {len(t)} rows")

    merged = pa.concat_tables(tables)
    n_total = len(merged)
    print(f"\nMerged: {n_total} total rows")

    # Compute stats on scored docs (non-null MCB)
    mcb_col = merged.column("longppl_mean_context_benefit").to_pylist()
    scored_mask = [v is not None for v in mcb_col]
    n_scored = sum(scored_mask)
    n_unscored = n_total - n_scored
    print(f"Scored: {n_scored}, Unscored (too short): {n_unscored}")

    # Distribution stats
    stats = {"source": args.source, "n_total": n_total, "n_scored": n_scored}
    for col_name in SCORE_COLUMNS:
        vals = [v for v in merged.column(col_name).to_pylist() if v is not None]
        if not vals:
            continue
        arr = np.array(vals)
        stats[col_name] = {
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "p10": round(float(np.percentile(arr, 10)), 6),
            "p25": round(float(np.percentile(arr, 25)), 6),
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p75": round(float(np.percentile(arr, 75)), 6),
            "p90": round(float(np.percentile(arr, 90)), 6),
            "max": round(float(np.max(arr)), 6),
        }

    # Write merged
    merged_path = args.output_dir.replace("gs://", "") + f"/longppl_all_{args.source}.parquet"
    with fs.open(merged_path, "wb") as f:
        pq.write_table(merged, f)
    print(f"Merged written to gs://{merged_path}")

    # Filter
    if args.filter_percentile > 0 and args.filter_metric in [c for c in merged.column_names]:
        filter_vals = merged.column(args.filter_metric).to_pylist()
        scored_vals = [v for v in filter_vals if v is not None]
        if scored_vals:
            threshold = float(np.percentile(scored_vals, args.filter_percentile))
            stats["filter"] = {
                "metric": args.filter_metric,
                "percentile": args.filter_percentile,
                "threshold": round(threshold, 6),
            }

            # Keep docs above threshold OR unscored (too short to evaluate)
            keep_mask = [
                (v is not None and v >= threshold) or v is None
                for v in filter_vals
            ]
            filtered = merged.filter(keep_mask)
            n_kept = len(filtered)
            n_dropped = n_total - n_kept
            stats["filter"]["n_kept"] = n_kept
            stats["filter"]["n_dropped"] = n_dropped
            stats["filter"]["pct_kept"] = round(n_kept / n_total * 100, 1)

            print(f"\nFilter: {args.filter_metric} >= {threshold:.4f} (p{args.filter_percentile})")
            print(f"  Kept: {n_kept}/{n_total} ({n_kept/n_total:.1%})")

            filtered_path = args.output_dir.replace("gs://", "") + f"/longppl_filtered_{args.source}.parquet"
            with fs.open(filtered_path, "wb") as f:
                pq.write_table(filtered, f)
            print(f"Filtered written to gs://{filtered_path}")

    # Write stats
    stats_path = args.output_dir.replace("gs://", "") + f"/longppl_stats_{args.source}.json"
    with fs.open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to gs://{stats_path}")


if __name__ == "__main__":
    main()
