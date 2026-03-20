"""
Bin ladder-scored docs into KTR tiers and sample text excerpts.

Reads all context ladder score parquets from a GCS prefix, bins docs into N tiers
on a chosen metric (default: ktr_4096v32768), samples K docs per tier, fetches
their raw tokens from the source tokenized parquets, decodes via tokenizer, and
prints formatted cards. Optionally writes JSONL output.

Usage:
    python scripts/analyze_ktr_tiers.py \\
        --gcs-scores-path gs://consus-dataproc/ocr/ia-ascm/scores/context_ladder_32k \\
        --gcs-text-path gs://consus-dataproc/ocr/ia-ascm/text/tokenized_fulldocs \\
        --n-samples 3 --text-chars 1000

    python scripts/analyze_ktr_tiers.py \\
        --gcs-scores-path gs://consus-dataproc/ocr/ia-ascm/scores/context_ladder_32k \\
        --gcs-text-path gs://consus-dataproc/ocr/ia-ascm/text/tokenized_fulldocs \\
        --n-samples 10 --output /tmp/ktr_tiers_ia_ascm.jsonl
"""

import argparse
import io
import json
import os
import subprocess
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def gcs_list(gcs_prefix, recursive=False):
    """List GCS paths under a prefix. Returns sorted list of paths."""
    cmd = ["gsutil", "ls"]
    if recursive:
        cmd.append("-r")
    cmd.append(gcs_prefix.rstrip("/") + "/")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR listing {gcs_prefix}: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    return sorted(lines)


def gcs_cat(gcs_path):
    """Read a GCS file and return raw bytes."""
    result = subprocess.run(["gsutil", "cat", gcs_path], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"gsutil cat failed for {gcs_path}: {result.stderr.decode().strip()}")
    return result.stdout


# ---------------------------------------------------------------------------
# Load all ladder score parquets
# ---------------------------------------------------------------------------

def load_scores(gcs_scores_path):
    """Read all file_*.parquet from gcs_scores_path and return concatenated DataFrame."""
    lines = gcs_list(gcs_scores_path)
    files = [l for l in lines if l.endswith(".parquet")]
    if not files:
        print(f"ERROR: no parquet files found under {gcs_scores_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} score parquet(s). Loading...", flush=True)

    dfs = []
    for i, f in enumerate(files):
        try:
            raw = gcs_cat(f)
        except RuntimeError as e:
            print(f"  WARNING: {e}, skipping", file=sys.stderr)
            continue
        dfs.append(pd.read_parquet(io.BytesIO(raw)))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(files)} files read ({sum(len(d) for d in dfs):,} docs)", flush=True)

    if not dfs:
        print("ERROR: no data loaded.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total scored docs: {len(df):,}", flush=True)
    return df


# ---------------------------------------------------------------------------
# Build basename -> full GCS path index for tokenized parquets
# ---------------------------------------------------------------------------

def build_text_index(gcs_text_path):
    """Recursively list gcs_text_path and return {basename: full_gcs_path} for .parquet files."""
    lines = gcs_list(gcs_text_path, recursive=True)
    index = {}
    for path in lines:
        if path.endswith(".parquet"):
            basename = os.path.basename(path)
            index[basename] = path
    if not index:
        print(f"ERROR: no parquet files found under {gcs_text_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Indexed {len(index)} tokenized parquet file(s).", flush=True)
    return index


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

def assign_tiers(df, metric, n_tiers):
    """Add a 'tier' column (0 = lowest, n_tiers-1 = highest) based on quantile cut."""
    if metric not in df.columns:
        available = [c for c in df.columns if any(
            c.startswith(p) for p in ("ktr_", "mcb_", "frac_pos_", "cb_p90_"))]
        print(f"ERROR: metric '{metric}' not found in scores.", file=sys.stderr)
        print(f"Available metric columns: {available}", file=sys.stderr)
        sys.exit(1)

    labels = list(range(n_tiers))
    df = df.copy()
    df["tier"] = pd.qcut(df[metric], q=n_tiers, labels=labels, duplicates="drop")
    return df


# ---------------------------------------------------------------------------
# Fetch tokens and decode
# ---------------------------------------------------------------------------

def fetch_rows(gcs_path, row_indices):
    """Read a single tokenized parquet from GCS and return rows at given indices."""
    raw = gcs_cat(gcs_path)
    parquet_df = pd.read_parquet(io.BytesIO(raw))
    return parquet_df.iloc[list(row_indices)]


def decode_window(tokens, score_window_start, score_window_size, tokenizer):
    """Extract and decode the scoring window from a token list."""
    start = int(score_window_start)
    end = start + int(score_window_size)
    window_tokens = tokens[start:end]
    return tokenizer.decode(window_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

TIER_LABELS = {0: "LOW", 1: "MED", 2: "HIGH"}


def tier_label(tier_idx, n_tiers):
    if n_tiers == 3:
        return TIER_LABELS.get(tier_idx, f"TIER{tier_idx}")
    return f"TIER{tier_idx}"


def print_card(row, text, metric, n_tiers, text_chars):
    tier_idx = int(row["tier"])
    label = tier_label(tier_idx, n_tiers)
    metric_val = row[metric]

    # Collect all pairwise metric values
    pair_metrics = {c: row[c] for c in row.index
                    if any(c.startswith(p) for p in ("ktr_", "mcb_", "frac_pos_", "cb_p90_"))}
    pair_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(pair_metrics.items()))

    print(f"\n{'='*72}")
    print(f"[{label}] {metric}={metric_val:.4f}   "
          f"source={row.get('source_file', '?')}  row={row.get('row_index', '?')}")
    print(f"  doc_len={int(row.get('doc_len', 0)):,}  "
          f"c_max={int(row.get('c_max', 0)):,}  "
          f"score_window_start={int(row.get('score_window_start', 0)):,}")
    if pair_str:
        print(f"  {pair_str}")
    print("---")
    excerpt = text[:text_chars]
    if len(text) > text_chars:
        excerpt += f"\n[... truncated at {text_chars} chars ...]"
    print(excerpt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bin ladder-scored docs into KTR tiers and sample text excerpts.")
    parser.add_argument("--gcs-scores-path", required=True,
                        help="GCS prefix with ladder score parquets (file_*.parquet)")
    parser.add_argument("--gcs-text-path", required=True,
                        help="GCS prefix for tokenized parquets (recursive glob)")
    parser.add_argument("--metric", default="ktr_4096v32768",
                        help="Column to tier on (default: ktr_4096v32768)")
    parser.add_argument("--n-tiers", type=int, default=3,
                        help="Number of tiers (default: 3)")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Docs to sample per tier (default: 5)")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B",
                        help="HF model name for tokenizer (default: meta-llama/Meta-Llama-3.1-8B)")
    parser.add_argument("--text-chars", type=int, default=2000,
                        help="Max chars to print per doc excerpt (default: 2000)")
    parser.add_argument("--output", default=None,
                        help="Optional JSONL path for all sampled docs with text")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    # 1. Load all scores
    df = load_scores(args.gcs_scores_path)

    # Validate required columns
    for col in ("source_file", "row_index", "score_window_start"):
        if col not in df.columns:
            print(f"ERROR: expected column '{col}' not found in scores. "
                  f"Columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    # Handle score_window_size: use column if present, else default 10240
    if "score_window_size" in df.columns:
        df["_window_size"] = df["score_window_size"].fillna(10240).astype(int)
    else:
        df["_window_size"] = 10240

    # 2. Assign tiers
    df = assign_tiers(df, args.metric, args.n_tiers)
    for t in range(args.n_tiers):
        count = (df["tier"] == t).sum()
        label = tier_label(t, args.n_tiers)
        vals = df.loc[df["tier"] == t, args.metric]
        print(f"  Tier {label}: {count:,} docs  "
              f"{args.metric} [{vals.min():.4f}, {vals.max():.4f}]")

    # 3. Sample per tier
    sampled = (df.groupby("tier", observed=True)
               .apply(lambda g: g.sample(n=min(args.n_samples, len(g)),
                                         random_state=args.seed))
               .reset_index(drop=True))
    print(f"\nSampled {len(sampled)} docs across {args.n_tiers} tiers.", flush=True)

    # 4. Build text index
    text_index = build_text_index(args.gcs_text_path)

    # 5. Load tokenizer
    print(f"Loading tokenizer: {args.model} ...", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 6. Group sampled docs by source_file to batch GCS reads
    output_records = []
    groups = sampled.groupby("source_file")
    n_groups = len(groups)
    print(f"Fetching tokens from {n_groups} unique source file(s)...", flush=True)

    for file_idx, (source_file, group) in enumerate(groups):
        if source_file not in text_index:
            print(f"  WARNING: '{source_file}' not found in text index, skipping {len(group)} doc(s)",
                  file=sys.stderr)
            continue

        gcs_path = text_index[source_file]
        print(f"  [{file_idx+1}/{n_groups}] Reading {source_file} ({len(group)} doc(s))...",
              flush=True)

        try:
            rows_df = fetch_rows(gcs_path, group["row_index"].tolist())
        except RuntimeError as e:
            print(f"  WARNING: {e}, skipping", file=sys.stderr)
            continue

        # Map row_index -> fetched row
        rows_by_idx = {int(idx): row for idx, row in zip(group["row_index"], rows_df.itertuples())}

        for _, score_row in group.iterrows():
            row_index = int(score_row["row_index"])
            text_row = rows_by_idx.get(row_index)
            if text_row is None:
                print(f"  WARNING: row_index {row_index} not found in fetched data, skipping",
                      file=sys.stderr)
                continue

            # Extract tokens
            if hasattr(text_row, "tokens"):
                tokens = list(text_row.tokens)
            else:
                print(f"  WARNING: no 'tokens' column in {source_file}, skipping row {row_index}",
                      file=sys.stderr)
                continue

            window_size = int(score_row["_window_size"])
            text = decode_window(tokens, score_row["score_window_start"], window_size, tokenizer)

            print_card(score_row, text, args.metric, args.n_tiers, args.text_chars)

            if args.output is not None:
                record = score_row.drop("_window_size").to_dict()
                record["tier_label"] = tier_label(int(score_row["tier"]), args.n_tiers)
                record["text_excerpt"] = text[:args.text_chars]
                output_records.append(record)

    # 7. Write JSONL
    if args.output is not None:
        with open(args.output, "w") as f:
            for rec in output_records:
                # Convert any non-serializable types
                clean = {}
                for k, v in rec.items():
                    try:
                        json.dumps(v)
                        clean[k] = v
                    except (TypeError, ValueError):
                        clean[k] = str(v)
                f.write(json.dumps(clean) + "\n")
        print(f"\nWrote {len(output_records)} records to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
