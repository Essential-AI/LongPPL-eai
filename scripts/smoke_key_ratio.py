"""
Smoke test: compute per-document key token ratio via LSD (Long-Short Difference).

Reads tokenized parquet docs from GCS, runs Llama-3.1-8B to identify key tokens
(tokens that benefit significantly from long context), and reports the key token
ratio per document. Optionally extracts key token text spans.

Usage:
    python scripts/smoke_key_ratio.py --n-docs 3
    python scripts/smoke_key_ratio.py --n-docs 10 --output results.json
    python scripts/smoke_key_ratio.py --n-docs 100 --output results.json --key-tokens-output key_tokens.jsonl
"""

import argparse
import json
import subprocess
import sys
import time

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_docs_from_gcs(gcs_path, n_docs, min_tokens, token_column):
    """Load tokenized documents from a GCS parquet dataset."""
    import gcsfs
    import pyarrow.parquet as pq

    fs = gcsfs.GCSFileSystem()

    # Discover partition files
    all_files = sorted(fs.ls(gcs_path))
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    if not parquet_files:
        # Maybe partitioned into subdirectories
        parquet_files = sorted(fs.glob(gcs_path.rstrip("/") + "/**/*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found under {gcs_path}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet file(s). Reading first partition: {parquet_files[0]}")

    with fs.open(parquet_files[0], "rb") as f:
        table = pq.read_table(f)

    # Print schema for discoverability
    print(f"\nParquet schema ({len(table.column_names)} columns):")
    for name, dtype in zip(table.column_names, table.schema.types):
        print(f"  {name}: {dtype}")
    print()

    if token_column not in table.column_names:
        print(f"ERROR: Expected token column '{token_column}' not found.")
        print(f"Available columns: {table.column_names}")
        print("Re-run with --token-column <name> to specify the correct column.")
        sys.exit(1)

    # Convert to Python lists for filtering
    docs = []
    col = table.column(token_column)
    for row_idx in range(len(table)):
        tokens = col[row_idx].as_py()
        if tokens is not None and len(tokens) >= min_tokens:
            docs.append({"doc_index": row_idx, "input_ids": tokens})
        if len(docs) >= n_docs:
            break

    if not docs:
        print(f"ERROR: No documents with >= {min_tokens} tokens found in first partition.")
        sys.exit(1)

    print(f"Selected {len(docs)} doc(s) with >= {min_tokens} tokens.\n")
    return docs


# ---------------------------------------------------------------------------
# Core computation — forked from longppl/longppl.py:find_key_token (lines 33-48)
# ---------------------------------------------------------------------------

def find_key_tokens_from_ids(
    input_ids,          # 1D tensor of token IDs (already on device)
    model,
    max_length,
    trunc_len=4096,
    sliding_window=1024,
    alpha=2.0,
    beta=-2.0,
    score_start=None,
):
    """
    Identify key tokens using LSD: tokens where the short-context loss is much
    higher than the full-context loss (indicating benefit from long context).

    Returns (key_positions, n_key, n_total, loss_full_all, loss_short_all) where
    key_positions is a list of absolute token positions, and loss_full_all /
    loss_short_all are 1-D numpy arrays of per-token losses over all scoreable tokens.
    """
    if score_start is None:
        score_start = trunc_len

    # Truncate and add batch dim
    input_ids = input_ids[:max_length].unsqueeze(0)
    seq_len = input_ids.shape[1]

    if seq_len <= trunc_len:
        print(f"  WARNING: doc length ({seq_len}) <= trunc_len ({trunc_len}), skipping")
        return [], 0, seq_len, np.array([]), np.array([])

    # Full-context forward pass
    with torch.no_grad():
        output_full = model(input_ids)

    loss_f = torch.nn.CrossEntropyLoss(reduction="none")
    key_positions = []

    # Collect per-token losses for all scoreable tokens
    loss_full_parts = []
    loss_short_parts = []

    with torch.no_grad():
        for start_token in range(score_start - trunc_len, seq_len - trunc_len, sliding_window):
            # Fix sliding_window mutation bug from original (line 36):
            # use a local variable instead of mutating the outer sliding_window
            sw = sliding_window
            if start_token + trunc_len + sw > seq_len:
                sw = seq_len - start_token - trunc_len
            if sw <= 0:
                break

            input_ids_short = input_ids[:, start_token : start_token + trunc_len + sw]
            output_short = model(input_ids_short)

            loss_full_window = loss_f(
                output_full.logits[0, start_token + trunc_len - 1 : start_token + trunc_len + sw - 1, :],
                input_ids[0, start_token + trunc_len : start_token + trunc_len + sw],
            )
            loss_short_window = loss_f(
                output_short.logits[0, trunc_len - 1 : trunc_len + sw - 1, :],
                input_ids_short[0, trunc_len : trunc_len + sw],
            )

            loss_full_parts.append(loss_full_window.cpu().float().numpy())
            loss_short_parts.append(loss_short_window.cpu().float().numpy())

            is_key = torch.logical_and(
                (loss_short_window - loss_full_window) > alpha,
                loss_full_window < (-1 * beta),
            )
            for j in range(is_key.shape[0]):
                if is_key[j]:
                    key_positions.append(start_token + trunc_len + j)

    # Concatenate all windows into contiguous arrays
    loss_full_all = np.concatenate(loss_full_parts) if loss_full_parts else np.array([])
    loss_short_all = np.concatenate(loss_short_parts) if loss_short_parts else np.array([])

    # Total scoreable tokens = those after score_start
    n_total = seq_len - score_start
    return key_positions, len(key_positions), n_total, loss_full_all, loss_short_all


# ---------------------------------------------------------------------------
# Key token text extraction
# ---------------------------------------------------------------------------

def extract_key_token_spans(input_ids_list, key_positions, tokenizer, context_window=5):
    """
    Decode key token positions back to text with surrounding context.

    Args:
        input_ids_list: flat list of token IDs for the document
        key_positions: list of absolute token positions that are key tokens
        tokenizer: HF tokenizer for decoding
        context_window: number of tokens of context on each side

    Returns:
        list of dicts with position, key_token_text, and context_text
    """
    if not key_positions:
        return []

    max_pos = len(input_ids_list)

    # Group adjacent key positions into contiguous spans to reduce output size
    spans = []
    sorted_positions = sorted(key_positions)
    span_start = sorted_positions[0]
    span_end = sorted_positions[0]

    for pos in sorted_positions[1:]:
        if pos <= span_end + 1:
            span_end = pos
        else:
            spans.append((span_start, span_end))
            span_start = pos
            span_end = pos
    spans.append((span_start, span_end))

    results = []
    for span_start, span_end in spans:
        # Key token text (just the key tokens in this span)
        key_ids = input_ids_list[span_start : span_end + 1]
        key_text = tokenizer.decode(key_ids, skip_special_tokens=True)

        # Context text (surrounding tokens for readability)
        ctx_start = max(0, span_start - context_window)
        ctx_end = min(max_pos, span_end + 1 + context_window)
        ctx_ids = input_ids_list[ctx_start : ctx_end]
        ctx_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)

        results.append({
            "span_start": span_start,
            "span_end": span_end,
            "n_key_tokens": span_end - span_start + 1,
            "key_text": key_text,
            "context_text": ctx_text,
        })

    return results


# ---------------------------------------------------------------------------
# Extended context-dependency metrics
# ---------------------------------------------------------------------------

def compute_extended_metrics(loss_full_all, loss_short_all, key_positions, trunc_len, n_total, alpha, beta, score_start=None):
    """
    Compute extended long-context dependency metrics from per-token losses.

    Args:
        loss_full_all: 1-D numpy array, full-context loss per scoreable token
        loss_short_all: 1-D numpy array, short-context loss per scoreable token
        key_positions: list of absolute token positions that are key tokens
        trunc_len: short context window size (for position-based analysis)
        n_total: total scoreable tokens
        alpha: LSD threshold
        beta: LCL threshold

    Returns:
        dict of metric name -> value (all rounded floats)
    """
    if len(loss_full_all) == 0 or len(loss_short_all) == 0:
        return {}

    context_benefit = loss_short_all - loss_full_all  # positive = long context helps

    metrics = {}

    # 1. Mean Context Benefit
    metrics["mean_context_benefit"] = round(float(np.mean(context_benefit)), 6)

    # 2. Median Context Benefit
    metrics["median_context_benefit"] = round(float(np.median(context_benefit)), 6)

    # 3. Context Benefit Std
    metrics["context_benefit_std"] = round(float(np.std(context_benefit)), 6)

    # 3b. Context Benefit Percentiles (captures right tail where key tokens live)
    metrics["context_benefit_p75"] = round(float(np.percentile(context_benefit, 75)), 6)
    metrics["context_benefit_p90"] = round(float(np.percentile(context_benefit, 90)), 6)
    metrics["context_benefit_p95"] = round(float(np.percentile(context_benefit, 95)), 6)

    # 4. Fraction with positive context benefit
    metrics["frac_positive_benefit"] = round(float(np.mean(context_benefit > 0)), 6)

    # 5. Weighted Context Benefit (continuous LongCE weight, averaged)
    # Clamp at 5.0 to match finetune.py's thre=5 (exp(5)~148, exp(20)~5e8 is unstable)
    clamped_benefit = np.clip(context_benefit, None, 5.0)
    metrics["weighted_context_benefit"] = round(float(np.mean(np.exp(clamped_benefit) - 1)), 6)

    # 6. Mean key token loss improvement (restricted to key tokens)
    if key_positions:
        # Key positions are absolute; scoreable tokens start at score_start.
        # The loss arrays are indexed 0..len-1 corresponding to windows in order.
        # Since windows tile the scoreable range without overlap in the non-overlapping
        # portions, we need to map absolute positions to array indices.
        _score_start = score_start if score_start is not None else trunc_len
        # The scoreable range covers positions [_score_start, _score_start + len(loss_full_all)).
        key_indices = [p - _score_start for p in key_positions]
        valid_key_indices = [i for i in key_indices if 0 <= i < len(context_benefit)]
        if valid_key_indices:
            metrics["mean_key_improvement"] = round(float(np.mean(context_benefit[valid_key_indices])), 6)
        else:
            metrics["mean_key_improvement"] = None
    else:
        metrics["mean_key_improvement"] = None

    # 7. Context benefit by position (split scoreable range into thirds)
    n = len(context_benefit)
    third = n // 3
    if third > 0:
        metrics["context_benefit_near"] = round(float(np.mean(context_benefit[:third])), 6)
        metrics["context_benefit_mid"] = round(float(np.mean(context_benefit[third:2*third])), 6)
        metrics["context_benefit_far"] = round(float(np.mean(context_benefit[2*third:])), 6)
    else:
        metrics["context_benefit_near"] = None
        metrics["context_benefit_mid"] = None
        metrics["context_benefit_far"] = None

    return metrics


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------

def compute_distribution_stats(ratios):
    """Compute percentile statistics for key token ratio distribution."""
    arr = np.array(ratios)
    return {
        "count": len(arr),
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name):
    """Load model with bfloat16 + SDPA attention (memory-efficient, works on ROCm)."""
    from transformers import AutoModelForCausalLM

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded. Device map: {model.hf_device_map}\n")
    return model


def load_tokenizer(model_name):
    """Load tokenizer for decoding key token positions back to text."""
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smoke test: per-document key token ratio via LSD"
    )
    parser.add_argument(
        "--gcs-path",
        type=str,
        default="gs://consus-dataproc/ocr/ia-ascm/text/tokenized_fulldocs",
        help="GCS path to tokenized parquet dataset",
    )
    parser.add_argument(
        "--token-column",
        type=str,
        default="tokens",
        help="Name of the column containing token ID lists",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Evaluator model name or path",
    )
    parser.add_argument("--n-docs", type=int, default=10, help="Number of documents to process")
    parser.add_argument(
        "--min-tokens", type=int, default=8192, help="Minimum token count to include a document"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2**15,  # 32768
        help="Truncate documents to this many tokens",
    )
    parser.add_argument("--trunc-len", type=int, default=4096, help="Short context window size")
    parser.add_argument("--sliding-window", type=int, default=1024, help="Sliding window size")
    parser.add_argument("--alpha", type=float, default=2.0, help="LSD threshold (loss discrepancy)")
    parser.add_argument("--beta", type=float, default=-2.0, help="LCL threshold (loss magnitude)")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON results")
    parser.add_argument(
        "--key-tokens-output",
        type=str,
        default=None,
        help="Path to write JSONL with key token text spans per document",
    )
    parser.add_argument(
        "--score-start", type=int, default=None,
        help="Start scoring at this absolute token position (default: trunc_len). "
             "Set to max_context_length to ensure all scored tokens have full context.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
        help="Number of context tokens on each side of key token spans",
    )
    parser.add_argument(
        "--max-spans-per-doc",
        type=int,
        default=50,
        help="Max key token spans to output per document (largest spans first)",
    )
    args = parser.parse_args()

    # Treat 0 as unset (for nox template compatibility where default is 0)
    if args.score_start is not None and args.score_start == 0:
        args.score_start = None

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

    # --- Load data ---
    docs = load_docs_from_gcs(args.gcs_path, args.n_docs, args.min_tokens, args.token_column)

    # --- Load model and tokenizer ---
    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model) if args.key_tokens_output else None

    # --- Process documents ---
    results = []
    key_tokens_records = []
    print(f"{'doc_index':>10}  {'key_ratio':>10}  {'n_key':>8}  {'n_total':>8}  {'doc_len':>8}  {'time_s':>8}  {'MCB':>11}")
    print("-" * 80)

    for doc in docs:
        input_ids = torch.tensor(doc["input_ids"], dtype=torch.long, device=model.device)
        doc_len = len(doc["input_ids"])

        t0 = time.time()
        try:
            key_positions, n_key, n_total, loss_full_all, loss_short_all = find_key_tokens_from_ids(
                input_ids,
                model,
                max_length=args.max_length,
                trunc_len=args.trunc_len,
                sliding_window=args.sliding_window,
                alpha=args.alpha,
                beta=args.beta,
                score_start=args.score_start,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM on doc {doc['doc_index']} ({doc_len} tokens), skipping")
            continue
        elapsed = time.time() - t0
        del input_ids
        torch.cuda.empty_cache()

        ratio = n_key / n_total if n_total > 0 else 0.0
        # Effective score_start for metadata
        effective_score_start = args.score_start if args.score_start is not None else args.trunc_len
        row = {
            "doc_index": doc["doc_index"],
            "score_start": effective_score_start,
            "key_token_ratio": round(ratio, 6),
            "n_key": n_key,
            "n_total": n_total,
            "doc_len": doc_len,
            "time_s": round(elapsed, 1),
        }

        # Compute extended metrics from per-token losses
        ext = compute_extended_metrics(
            loss_full_all, loss_short_all, key_positions,
            trunc_len=args.trunc_len, n_total=n_total,
            alpha=args.alpha, beta=args.beta,
            score_start=args.score_start,
        )
        row.update(ext)

        results.append(row)
        mcb_str = f"{row.get('mean_context_benefit', 0):>8.3f}" if row.get('mean_context_benefit') is not None else "     N/A"
        print(f"{row['doc_index']:>10}  {row['key_token_ratio']:>10.4f}  {row['n_key']:>8}  {row['n_total']:>8}  {row['doc_len']:>8}  {row['time_s']:>8.1f}  MCB={mcb_str}")

        # Extract key token text spans if requested
        if tokenizer is not None and key_positions:
            # Use only up to max_length tokens (same truncation as the model saw)
            truncated_ids = doc["input_ids"][:args.max_length]
            spans = extract_key_token_spans(
                truncated_ids, key_positions, tokenizer, args.context_window
            )
            # Keep largest spans first, limited to max_spans_per_doc
            spans.sort(key=lambda s: s["n_key_tokens"], reverse=True)
            spans = spans[:args.max_spans_per_doc]
            key_tokens_records.append({
                "doc_index": doc["doc_index"],
                "key_token_ratio": round(ratio, 6),
                "n_key": n_key,
                "n_spans": len(spans),
                "key_token_spans": spans,
            })

    # --- Summary ---
    ratios = [r["key_token_ratio"] for r in results]
    if not ratios:
        print("\nNo documents were scored successfully.")
        return
    print(f"\nSummary: mean ratio = {np.mean(ratios):.4f}, "
          f"min = {np.min(ratios):.4f}, max = {np.max(ratios):.4f}")

    # Extended metrics summary
    ext_metric_names = [
        "mean_context_benefit", "median_context_benefit", "context_benefit_std",
        "context_benefit_p75", "context_benefit_p90", "context_benefit_p95",
        "frac_positive_benefit", "weighted_context_benefit", "mean_key_improvement",
        "context_benefit_near", "context_benefit_mid", "context_benefit_far",
    ]
    print("\nExtended metrics (mean across docs):")
    for m in ext_metric_names:
        vals = [r[m] for r in results if r.get(m) is not None]
        if vals:
            print(f"  {m}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # Distribution statistics (useful when n_docs is large)
    if len(ratios) >= 5:
        stats = compute_distribution_stats(ratios)
        print(f"\nKey token ratio distribution (n={stats['count']}):")
        print(f"  p10={stats['p10']:.4f}  p25={stats['p25']:.4f}  p50={stats['p50']:.4f}  "
              f"p75={stats['p75']:.4f}  p90={stats['p90']:.4f}")
        print(f"  mean={stats['mean']:.4f}  std={stats['std']:.4f}")

        # Filtering thresholds: what fraction of docs would be kept at various ratio cutoffs
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        arr = np.array(ratios)
        print(f"\n  Fraction of docs above threshold:")
        for t in thresholds:
            frac = (arr >= t).mean()
            print(f"    ratio >= {t:.3f}: {frac:.1%} ({(arr >= t).sum()}/{len(arr)})")

        # Extended metric distributions
        ext_stats = {}
        for m in ext_metric_names:
            vals = [r[m] for r in results if r.get(m) is not None]
            if len(vals) >= 5:
                ext_stats[m] = compute_distribution_stats(vals)
    else:
        stats = None
        ext_stats = {}

    # --- Write output ---
    if args.output:
        output_data = {
            "results": results,
        }
        if stats:
            output_data["distribution"] = stats
        if ext_stats:
            output_data["extended_distributions"] = ext_stats
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to {args.output}")

    # --- Write key tokens JSONL ---
    if args.key_tokens_output and key_tokens_records:
        with open(args.key_tokens_output, "w") as f:
            for record in key_tokens_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Key token spans written to {args.key_tokens_output}")


if __name__ == "__main__":
    main()
