"""
Summarize context ladder scoring results from GCS.

Reads all output parquets from a GCS prefix, concatenates them, and writes
pandas describe() stats to a CSV.

Usage:
    python scripts/summarize_ladder_results.py \
        --gcs-path gs://consus-dataproc/ocr/ia-ascm/scores/context_ladder_32k \
        --output /tmp/ia_ascm_ladder_summary.csv
"""

import argparse
import io
import os
import subprocess
import sys

import pandas as pd


def _decile_path(output_path):
    base, ext = os.path.splitext(output_path)
    return base + "_deciles" + ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-path", required=True,
                        help="GCS prefix containing file_*.parquet outputs")
    parser.add_argument("--output", default=None,
                        help="Local path to write summary CSV")
    parser.add_argument("--count-only", action="store_true",
                        help="Just print total doc count, skip describe stats")
    args = parser.parse_args()

    if not args.count_only and args.output is None:
        parser.error("--output is required unless --count-only is set")

    # List all parquet files
    result = subprocess.run(
        ["gsutil", "ls", args.gcs_path.rstrip("/") + "/"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR listing {args.gcs_path}: {result.stderr.strip()}")
        sys.exit(1)

    files = sorted(l.strip() for l in result.stdout.strip().split("\n")
                   if l.strip().endswith(".parquet"))
    print(f"Found {len(files)} parquet files.")

    # Read and concatenate
    dfs = []
    total = 0
    files_read = 0
    for i, f in enumerate(files):
        r = subprocess.run(["gsutil", "cat", f], capture_output=True)
        if r.returncode != 0:
            print(f"  WARNING: could not read {f}, skipping")
            continue
        files_read += 1
        if args.count_only:
            df_f = pd.read_parquet(io.BytesIO(r.stdout), columns=["row_index"])
            total += len(df_f)
        else:
            dfs.append(pd.read_parquet(io.BytesIO(r.stdout)))
        if (i + 1) % 25 == 0:
            docs_so_far = total if args.count_only else sum(len(d) for d in dfs)
            print(f"  Read {i+1}/{len(files)} files ({docs_so_far:,} docs so far)")

    if args.count_only:
        print(f"Files with data: {files_read} / {len(files)}")
        print(f"Total docs scored: {total:,}")
        return

    if not dfs:
        print("ERROR: no data read.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal docs: {len(df):,}")
    print(f"Columns: {list(df.columns)}\n")

    # Summarize: doc_len + all metric columns
    metric_cols = ["doc_len"] + [c for c in df.columns
                                  if any(c.startswith(p) for p in
                                         ("mcb_", "ktr_", "frac_pos_", "cb_p90_"))]
    summary = df[metric_cols].describe()
    print(summary.to_string())

    summary.to_csv(args.output)
    print(f"\nSummary written to {args.output}")

    # Decile analysis on ktr_4096v32768
    ktr_col = "ktr_4096v32768"
    if ktr_col not in df.columns:
        print(f"\nColumn {ktr_col!r} not found; skipping decile analysis.")
        return

    df_nan = df[df[ktr_col].isna()]
    df_valid = df[df[ktr_col].notna()].copy()

    print(f"\n--- Too-short docs (ktr_4096v32768 is NaN) ---")
    print(f"Count: {len(df_nan):,}")
    if len(df_nan) > 0:
        print(f"doc_len mean:   {df_nan['doc_len'].mean():.1f}")
        print(f"doc_len median: {df_nan['doc_len'].median():.1f}")

    if len(df_valid) == 0:
        print("\nNo valid docs for decile analysis.")
        return

    df_valid["_decile"] = pd.qcut(df_valid[ktr_col], q=10, labels=False, duplicates="drop")

    count_col = "row_index" if "row_index" in df_valid.columns else df_valid.columns[0]
    agg_dict = {
        "n_docs": (count_col, "count"),
        "ktr_min": (ktr_col, "min"),
        "ktr_max": (ktr_col, "max"),
        "doc_len_mean": ("doc_len", "mean"),
        "doc_len_median": ("doc_len", "median"),
        "ktr_mean": (ktr_col, "mean"),
    }
    for col in ["mcb_4096v32768", "frac_pos_4096v32768"]:
        if col in df_valid.columns:
            agg_dict[col + "_mean"] = (col, "mean")

    decile_df = df_valid.groupby("_decile").agg(**agg_dict).reset_index(drop=True)
    decile_df.index = [f"D{i+1}" for i in range(len(decile_df))]

    print(f"\n--- Decile breakdown on {ktr_col} (n={len(df_valid):,} docs) ---")
    print(decile_df.to_string(float_format=lambda x: f"{x:.4f}"))

    decile_path = _decile_path(args.output)
    decile_df.to_csv(decile_path)
    print(f"\nDecile table written to {decile_path}")


if __name__ == "__main__":
    main()
