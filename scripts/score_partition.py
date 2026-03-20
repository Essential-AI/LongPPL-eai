"""
Scalable LongPPL scoring pipeline: score a single parquet partition.

Reads one parquet file, scores every document that meets min_tokens, and writes
a parquet file with the original columns plus LongPPL metric columns. Documents
below min_tokens get null scores.

Designed to be parallelized: one GPU job per partition file. The nox launcher
generates one CSV row per partition, and each job writes its scored output to a
mirrored path under a scores/ prefix.

Usage:
    python scripts/score_partition.py \
        --input gs://consus-dataproc/ocr/arxiv/text/tokenized_fulldocs/batch_.../part-00000.parquet \
        --output gs://consus-dataproc/ocr/arxiv/scores/longppl/part-00000.parquet \
        --model meta-llama/Meta-Llama-3.1-8B

    # Process a range of rows within a partition (for splitting large files)
    python scripts/score_partition.py \
        --input gs://...part-00000.parquet \
        --output gs://...part-00000_rows0-500.parquet \
        --row-start 0 --row-end 500
"""

import argparse
import json
import sys
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Scoring (reused from smoke_key_ratio.py, streamlined for throughput)
# ---------------------------------------------------------------------------

def score_document(input_ids, model, max_length, trunc_len, sliding_window):
    """
    Score a single document. Returns dict of metric values, or None if too short.
    """
    input_ids = input_ids[:max_length].unsqueeze(0)
    seq_len = input_ids.shape[1]

    if seq_len <= trunc_len:
        return None

    with torch.no_grad():
        output_full = model(input_ids)

    loss_f = torch.nn.CrossEntropyLoss(reduction="none")
    loss_full_parts = []
    loss_short_parts = []
    n_key = 0

    alpha = 2.0
    beta = -2.0

    with torch.no_grad():
        for start_token in range(0, seq_len - trunc_len, sliding_window):
            sw = sliding_window
            if start_token + trunc_len + sw > seq_len:
                sw = seq_len - start_token - trunc_len
            if sw <= 0:
                break

            input_ids_short = input_ids[:, start_token : start_token + trunc_len + sw]
            output_short = model(input_ids_short)

            lf = loss_f(
                output_full.logits[0, start_token + trunc_len - 1 : start_token + trunc_len + sw - 1, :],
                input_ids[0, start_token + trunc_len : start_token + trunc_len + sw],
            )
            ls = loss_f(
                output_short.logits[0, trunc_len - 1 : trunc_len + sw - 1, :],
                input_ids_short[0, trunc_len : trunc_len + sw],
            )

            loss_full_parts.append(lf.cpu().float().numpy())
            loss_short_parts.append(ls.cpu().float().numpy())

            is_key = torch.logical_and((ls - lf) > alpha, lf < (-1 * beta))
            n_key += int(is_key.sum().item())

    loss_full_all = np.concatenate(loss_full_parts) if loss_full_parts else np.array([])
    loss_short_all = np.concatenate(loss_short_parts) if loss_short_parts else np.array([])

    if len(loss_full_all) == 0:
        return None

    cb = loss_short_all - loss_full_all
    n_total = seq_len - trunc_len
    n = len(cb)
    third = n // 3

    metrics = {
        "longppl_key_token_ratio": round(float(n_key / n_total), 6) if n_total > 0 else 0.0,
        "longppl_n_key": n_key,
        "longppl_n_total": n_total,
        "longppl_mean_context_benefit": round(float(np.mean(cb)), 6),
        "longppl_median_context_benefit": round(float(np.median(cb)), 6),
        "longppl_context_benefit_std": round(float(np.std(cb)), 6),
        "longppl_context_benefit_p75": round(float(np.percentile(cb, 75)), 6),
        "longppl_context_benefit_p90": round(float(np.percentile(cb, 90)), 6),
        "longppl_context_benefit_p95": round(float(np.percentile(cb, 95)), 6),
        "longppl_frac_positive_benefit": round(float(np.mean(cb > 0)), 6),
        "longppl_weighted_context_benefit": round(float(np.mean(np.exp(np.clip(cb, None, 5.0)) - 1)), 6),
        "longppl_context_benefit_near": round(float(np.mean(cb[:third])), 6) if third > 0 else None,
        "longppl_context_benefit_mid": round(float(np.mean(cb[third:2*third])), 6) if third > 0 else None,
        "longppl_context_benefit_far": round(float(np.mean(cb[2*third:])), 6) if third > 0 else None,
    }
    return metrics


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_partition(input_path, token_column, row_start, row_end):
    """Read a parquet partition from GCS or local. Returns pyarrow table."""
    import pyarrow.parquet as pq

    if input_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(input_path, "rb") as f:
            table = pq.read_table(f)
    else:
        table = pq.read_table(input_path)

    if token_column not in table.column_names:
        print(f"ERROR: column '{token_column}' not found. Available: {table.column_names}")
        sys.exit(1)

    if row_end is not None:
        table = table.slice(row_start, row_end - row_start)
    elif row_start > 0:
        table = table.slice(row_start)

    return table


def write_partition(table, output_path):
    """Write scored parquet to GCS or local."""
    import pyarrow.parquet as pq

    if output_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(output_path, "wb") as f:
            pq.write_table(table, f)
    else:
        pq.write_table(table, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score a single parquet partition with LongPPL metrics")
    parser.add_argument("--input", required=True, help="Input parquet path (GCS or local)")
    parser.add_argument("--output", required=True, help="Output parquet path (GCS or local)")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--token-column", default="tokens")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--trunc-len", type=int, default=4096)
    parser.add_argument("--sliding-window", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=8192)
    parser.add_argument("--row-start", type=int, default=0, help="Start row index (inclusive)")
    parser.add_argument("--row-end", type=int, default=None, help="End row index (exclusive)")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N docs")
    args = parser.parse_args()

    import pyarrow as pa

    print(f"Reading {args.input} ...")
    table = read_partition(args.input, args.token_column, args.row_start, args.row_end)
    n_rows = len(table)
    print(f"  {n_rows} rows, columns: {table.column_names}")

    # Load model
    from transformers import AutoModelForCausalLM
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded. Device map: {model.hf_device_map}")

    # Score columns (initialized as None)
    score_columns = [
        "longppl_key_token_ratio", "longppl_n_key", "longppl_n_total",
        "longppl_mean_context_benefit", "longppl_median_context_benefit",
        "longppl_context_benefit_std",
        "longppl_context_benefit_p75", "longppl_context_benefit_p90",
        "longppl_context_benefit_p95",
        "longppl_frac_positive_benefit",
        "longppl_weighted_context_benefit",
        "longppl_context_benefit_near", "longppl_context_benefit_mid",
        "longppl_context_benefit_far",
    ]
    scores = {col: [None] * n_rows for col in score_columns}

    token_col = table.column(args.token_column)
    t_total = time.time()
    n_scored = 0
    n_skipped = 0

    n_oom = 0
    for i in range(n_rows):
        tokens = token_col[i].as_py()
        if tokens is None or len(tokens) < args.min_tokens:
            n_skipped += 1
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long, device=model.device)
        t0 = time.time()
        try:
            metrics = score_document(
                input_ids, model, args.max_length, args.trunc_len, args.sliding_window
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_oom += 1
            n_skipped += 1
            print(f"  [{i+1}/{n_rows}] OOM on doc with {len(tokens)} tokens, skipping")
            continue
        elapsed = time.time() - t0

        # Free GPU memory between docs
        del input_ids
        torch.cuda.empty_cache()

        if metrics is not None:
            for col in score_columns:
                scores[col][i] = metrics.get(col)
            n_scored += 1

            if n_scored % args.progress_every == 0:
                rate = n_scored / (time.time() - t_total)
                remaining = (n_rows - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{n_rows}] scored={n_scored} skipped={n_skipped} "
                      f"rate={rate:.1f} docs/min  ETA={remaining/60:.0f}min  "
                      f"last_MCB={metrics['longppl_mean_context_benefit']:.4f}")
        else:
            n_skipped += 1

    elapsed_total = time.time() - t_total
    print(f"\nDone: {n_scored} scored, {n_skipped} skipped ({n_oom} OOM), {elapsed_total/60:.1f} min total")

    # Add score columns to table
    for col in score_columns:
        if col in ("longppl_n_key", "longppl_n_total"):
            arr = pa.array(scores[col], type=pa.int32())
        else:
            arr = pa.array(scores[col], type=pa.float32())
        table = table.append_column(col, arr)

    # Write output
    print(f"Writing {args.output} ...")
    write_partition(table, args.output)
    print(f"Written {n_rows} rows with {len(score_columns)} score columns.")

    # Summary stats for scored docs
    mcb_vals = [v for v in scores["longppl_mean_context_benefit"] if v is not None]
    if mcb_vals:
        mcb = np.array(mcb_vals)
        print(f"\nMCB summary: mean={np.mean(mcb):.4f} std={np.std(mcb):.4f} "
              f"min={np.min(mcb):.4f} max={np.max(mcb):.4f}")
    ktr_vals = [v for v in scores["longppl_key_token_ratio"] if v is not None]
    if ktr_vals:
        ktr = np.array(ktr_vals)
        print(f"KTR summary: mean={np.mean(ktr):.4f} std={np.std(ktr):.4f}")


if __name__ == "__main__":
    main()
