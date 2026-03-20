"""
Multi-context-length LongPPL scoring: compute per-token loss at multiple context
lengths on the same scoring window, enabling pairwise context benefit analysis.

For each document, picks a fixed scoring window of W tokens, then runs one forward
pass per context length c in {4K, 8K, 16K, 32K, 64K, 128K} on input[P-c : P+W],
where P = c_max (the largest context that fits). Every context length scores the
exact same tokens -- only the amount of preceding context changes.

Each doc uses an adaptive ladder: only context lengths where c + W <= doc_length.

Output: flat parquet with one row per doc and columns for each pairwise metric,
suitable for joining back to the source parquet by (source_file, row_index).

Checkpointing: uploads partial results to GCS every --checkpoint-interval docs.
On restart, downloads existing results and skips already-scored docs, so
preemption only loses work since the last checkpoint.

Usage:
    python scripts/context_ladder.py \
        --gcs-path gs://consus-dataproc/ocr/arxiv/text/tokenized_fulldocs \
        --n-docs 10 --min-tokens 14336 --score-window 10240 \
        --output /tmp/test_ladder.parquet \
        --output-gcs gs://consus-dataproc/ocr/arxiv/scores/ladder_0.parquet
"""

import argparse
import os
import subprocess
import sys
import time

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_docs_from_gcs(gcs_path, n_docs, min_tokens, token_column,
                       doc_offset=0, file_index=None):
    """Load tokenized documents from a GCS parquet dataset.

    Args:
        file_index: If set, only read this parquet file (0-indexed) and process
            all eligible docs in it. Much faster than doc_offset for multi-file
            datasets since it skips scanning earlier files.

    Returns list of dicts with keys:
        source_file: parquet filename (basename, not full path)
        row_index: row index within that parquet file
        input_ids: list of token IDs
        doc_len: number of tokens in the full document
    """
    import gcsfs
    import pyarrow.parquet as pq

    fs = gcsfs.GCSFileSystem()

    all_files = sorted(fs.ls(gcs_path))
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    if not parquet_files:
        parquet_files = sorted(fs.glob(gcs_path.rstrip("/") + "/**/*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found under {gcs_path}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet file(s).")

    # If file_index is set, only process that one file
    if file_index is not None:
        if file_index >= len(parquet_files):
            print(f"ERROR: --file-index {file_index} but only {len(parquet_files)} files found.")
            sys.exit(1)
        parquet_files = [parquet_files[file_index]]
        print(f"Using file index {file_index}: {parquet_files[0]}")

    docs = []
    skipped = 0
    for pf in parquet_files:
        if n_docs > 0 and len(docs) >= n_docs:
            break
        basename = os.path.basename(pf)
        print(f"  Reading: {pf}")
        with fs.open(pf, "rb") as f:
            # Only read the columns we need — avoids deserializing text/metadata
            all_col_names = pq.read_schema(f).names
        with fs.open(pf, "rb") as f:
            needed = [token_column] + (["token_len"] if "token_len" in all_col_names else [])
            table = pq.read_table(f, columns=needed)

        if token_column not in table.column_names:
            print(f"ERROR: Expected token column '{token_column}' not found.")
            print(f"Available columns: {table.column_names}")
            sys.exit(1)

        col = table.column(token_column)
        len_col = table.column("token_len").to_pylist() if "token_len" in table.column_names else None

        for row_idx in range(len(table)):
            doc_len = len_col[row_idx] if len_col is not None else None
            if doc_len is not None and doc_len < min_tokens:
                continue
            tokens = col[row_idx].as_py()
            if tokens is None or len(tokens) < min_tokens:
                continue
            if skipped < doc_offset:
                skipped += 1
                continue
            docs.append({
                "source_file": basename,
                "row_index": row_idx,
                "input_ids": tokens,
                "doc_len": doc_len if doc_len is not None else len(tokens),
            })
            if n_docs > 0 and len(docs) >= n_docs:
                break

    if not docs:
        print(f"ERROR: No documents with >= {min_tokens} tokens found.")
        sys.exit(1)

    print(f"Selected {len(docs)} doc(s) with >= {min_tokens} tokens.\n")
    return docs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name):
    """Load model with bfloat16 + SDPA on a single GPU."""
    from transformers import AutoModelForCausalLM

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded on GPU 0.\n")
    return model


# ---------------------------------------------------------------------------
# Core: score one doc at multiple context lengths
# ---------------------------------------------------------------------------

def score_doc_at_context_lengths(input_ids, model, context_lengths, score_window,
                                 lm_head_weight, loss_f):
    """
    Run one forward pass per context length, return per-token losses.

    Uses model.model() to get hidden states, slices to the scoring window,
    then a single fused lm_head matmul (10K x vocab — ~2.5GB, fine on 192GB).

    lm_head_weight and loss_f are passed in (hoisted out of the per-doc loop).

    Returns:
        (losses, P, ladder) where:
            losses: dict mapping context_length -> numpy array of shape (W,)
            P: scoring window start position (== c_max)
            ladder: list of context lengths used
    """
    L = len(input_ids)
    valid = [c for c in context_lengths if c + score_window <= L]
    if not valid:
        return None, None, None

    c_max = max(valid)
    P = c_max
    ladder = sorted(c for c in context_lengths if c <= c_max)

    losses = {}

    for c in ladder:
        start = P - c
        end = P + score_window
        input_slice = input_ids[start:end].unsqueeze(0)

        with torch.no_grad():
            hidden = model.model(input_slice).last_hidden_state
            scoring_hidden = hidden[0, c - 1 : c + score_window - 1]
            targets = input_slice[0, c : c + score_window]
            del hidden

            logits = scoring_hidden @ lm_head_weight.T
            loss = loss_f(logits, targets)
            del scoring_hidden, logits

        losses[c] = loss.cpu().float().numpy()
        del loss

    return losses, P, ladder


# ---------------------------------------------------------------------------
# Pairwise metrics — returns flat dict with column-name keys
# ---------------------------------------------------------------------------

def compute_pairwise_flat(losses, alpha, beta):
    """From per-context-length loss arrays, compute metrics for all pairs.

    Returns a flat dict like:
        {"mcb_4096v8192": 0.021, "ktr_4096v8192": 0.002, ...}
    """
    row = {}
    context_lengths = sorted(losses.keys())
    neg_beta = -1 * beta

    for i, c_short in enumerate(context_lengths):
        for c_long in context_lengths[i + 1:]:
            cb = losses[c_short] - losses[c_long]
            pair = f"{c_short}v{c_long}"
            is_key = (cb > alpha) & (losses[c_long] < neg_beta)
            row[f"mcb_{pair}"] = round(float(np.mean(cb)), 6)
            row[f"ktr_{pair}"] = round(float(np.mean(is_key)), 6)
            row[f"frac_pos_{pair}"] = round(float(np.mean(cb > 0)), 6)
            row[f"cb_p90_{pair}"] = round(float(np.percentile(cb, 90)), 6)
    return row


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_existing_results(output_gcs, local_path):
    """Download existing results from GCS and return set of already-scored (source_file, row_index)."""
    import pandas as pd

    done = set()
    try:
        result = subprocess.run(
            ["gsutil", "cp", output_gcs, local_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and os.path.exists(local_path):
            df = pd.read_parquet(local_path)
            done = set(zip(df["source_file"], df["row_index"].astype(int)))
            print(f"Resuming: loaded {len(done)} already-scored docs from {output_gcs}")
            return df.to_dict("records"), done
    except Exception as e:
        print(f"No existing results to resume from: {e}")
    return [], done


def checkpoint(rows, local_path, output_gcs):
    """Write rows to local parquet and upload to GCS."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_parquet(local_path, index=False)
    result = subprocess.run(
        ["gsutil", "-q", "cp", local_path, output_gcs],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  WARNING: GCS upload failed: {result.stderr.strip()}")
    else:
        print(f"  Checkpoint: {len(rows)} rows uploaded to {output_gcs}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-context-length LongPPL scoring"
    )
    parser.add_argument("--gcs-path", type=str, required=True)
    parser.add_argument("--token-column", type=str, default="tokens")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--n-docs", type=int, default=0,
                        help="Max docs to process (0 = all eligible docs)")
    parser.add_argument("--min-tokens", type=int, default=14336)
    parser.add_argument("--score-window", type=int, default=10240)
    parser.add_argument(
        "--context-lengths", type=str, default="4096,8192,16384,32768,65536,131072",
    )
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=-2.0)
    parser.add_argument("--output", type=str, default="/tmp/ladder_results.parquet",
                        help="Local output path (.parquet)")
    parser.add_argument("--output-gcs", type=str, default=None,
                        help="GCS path for checkpointing and final upload")
    parser.add_argument("--doc-offset", type=int, default=0)
    parser.add_argument("--file-index", type=int, default=None,
                        help="Process only this parquet file (0-indexed). "
                             "Much faster than --doc-offset for multi-file datasets.")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Upload partial results to GCS every N docs")
    args = parser.parse_args()

    context_lengths = sorted(int(x) for x in args.context_lengths.split(","))

    # --- GPU info ---
    print("=" * 60)
    print("GPU INFO")
    print("=" * 60)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5,
        )
        print(result.stdout.strip())
    except Exception as e:
        print(f"Could not query GPU: {e}")
    print("=" * 60 + "\n")

    # --- Resume from checkpoint ---
    rows = []
    done = set()
    if args.output_gcs:
        rows, done = load_existing_results(args.output_gcs, args.output)

    # --- Load data ---
    docs = load_docs_from_gcs(
        args.gcs_path, args.n_docs, args.min_tokens,
        args.token_column, doc_offset=args.doc_offset,
        file_index=args.file_index,
    )

    # Filter out already-scored docs
    if done:
        before = len(docs)
        docs = [d for d in docs if (d["source_file"], d["row_index"]) not in done]
        print(f"Skipping {before - len(docs)} already-scored docs, {len(docs)} remaining.\n")

    if not docs:
        print("All docs already scored. Nothing to do.")
        return

    # --- Load model ---
    model = load_model(args.model)

    # --- Warmup ---
    print("Warmup forward pass...", end=" ", flush=True)
    dummy = torch.zeros(1, 1024, dtype=torch.long, device=model.device)
    with torch.no_grad():
        model(dummy)
    del dummy
    torch.cuda.empty_cache()
    print("done.\n")

    # --- VRAM check ---
    # If another process has consumed most of the GPU memory, every doc will OOM
    # and the job will exit 0 ("No new documents were scored") — looking like success
    # to Kubernetes, which won't retry. Exit 1 here so maxRestarts reschedules us
    # on a different node.
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    print(f"GPU memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total after model load")
    if free_gb < 50:
        print(f"ERROR: Insufficient free GPU memory ({free_gb:.1f}GB < 50GB threshold). "
              f"Exiting with error so Kubernetes will reschedule on a different node.")
        sys.exit(1)

    # --- Hoist per-doc constants out of the loop ---
    lm_head_weight = model.lm_head.weight
    loss_f = torch.nn.CrossEntropyLoss(reduction="none")
    max_needed = max(context_lengths) + args.score_window

    # --- Process documents ---
    new_count = 0
    print(f"{'doc':>5}  {'file':>30}  {'row':>6}  {'len':>8}  {'c_max':>7}  {'time_s':>7}")
    print("-" * 75)

    total_t0 = time.time()
    for doc in docs:
        doc_len = doc["doc_len"]
        input_ids = torch.tensor(doc["input_ids"][:max_needed], dtype=torch.long, device=model.device)

        t0 = time.time()
        try:
            losses, P, ladder = score_doc_at_context_lengths(
                input_ids, model, context_lengths, args.score_window,
                lm_head_weight, loss_f,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM on {doc['source_file']}:{doc['row_index']} ({doc_len} tokens), skipping")
            continue
        elapsed = time.time() - t0

        del input_ids
        torch.cuda.empty_cache()

        if losses is None:
            print(f"  {doc['source_file']}:{doc['row_index']} ({doc_len} tokens) too short, skipping")
            continue

        pairwise = compute_pairwise_flat(losses, args.alpha, args.beta)

        row = {
            "source_file": doc["source_file"],
            "row_index": doc["row_index"],
            "doc_len": doc_len,
            "c_max": P,
            "score_window_start": P,
            "time_s": round(elapsed, 2),
        }
        row.update(pairwise)
        rows.append(row)
        new_count += 1

        print(f"{len(rows):>5}  {doc['source_file']:>30}  {doc['row_index']:>6}  "
              f"{doc_len:>8}  {P:>7}  {elapsed:>7.1f}")

        # Periodic checkpoint
        if args.output_gcs and new_count % args.checkpoint_interval == 0:
            checkpoint(rows, args.output, args.output_gcs)

    total_elapsed = time.time() - total_t0

    if new_count == 0:
        print("\nNo new documents were scored.")
        sys.exit(1)

    print(f"\nTotal: {total_elapsed:.1f}s for {new_count} new docs "
          f"({total_elapsed / new_count:.1f}s/doc avg)")

    # --- Print aggregate ---
    import pandas as pd
    df = pd.DataFrame(rows)
    mcb_cols = [c for c in df.columns if c.startswith("mcb_")]
    if mcb_cols:
        print(f"\n{'='*60}")
        print(f"AGGREGATE ({len(df)} docs total)")
        print(f"{'='*60}")
        for col in sorted(mcb_cols):
            pair = col.replace("mcb_", "")
            ktr_col = f"ktr_{pair}"
            fp_col = f"frac_pos_{pair}"
            n = df[col].notna().sum()
            print(f"  {pair}: n={n}  MCB={df[col].mean():.4f}  "
                  f"KTR={df[ktr_col].mean():.4f}  FracPos={df[fp_col].mean():.3f}")

    # --- Final write ---
    df.to_parquet(args.output, index=False)
    print(f"\nResults written to {args.output} ({len(df)} rows, {len(df.columns)} columns)")

    if args.output_gcs:
        checkpoint(rows, args.output, args.output_gcs)
        print("Final upload complete.")


if __name__ == "__main__":
    main()
