"""
Multi-context-length LongPPL scoring using vLLM for fast batch inference.

Uses vLLM's offline LLM API with prompt_logprobs to compute per-token
log-likelihoods at multiple context lengths. Submits docs one at a time
with chunked prefill to control memory usage.

Usage:
    python scripts/context_ladder_vllm.py \
        --gcs-path gs://consus-dataproc/ocr/arxiv/text/tokenized_fulldocs \
        --n-docs 10 --min-tokens 50000 --score-window 10240 \
        --context-lengths 4096,16384,32768 \
        --output /tmp/test_ladder.json
"""

import argparse
import json
import subprocess
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_docs_from_gcs(gcs_path, n_docs, min_tokens, token_column, doc_offset=0):
    """Load tokenized documents from a GCS parquet dataset."""
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
    docs = []
    skipped = 0
    for pf in parquet_files:
        if len(docs) >= n_docs:
            break
        print(f"  Reading: {pf}")
        with fs.open(pf, "rb") as f:
            table = pq.read_table(f)
        if token_column not in table.column_names:
            print(f"ERROR: column '{token_column}' not found. Available: {table.column_names}")
            sys.exit(1)
        col = table.column(token_column)
        for row_idx in range(len(table)):
            tokens = col[row_idx].as_py()
            if tokens is None or len(tokens) < min_tokens:
                continue
            if skipped < doc_offset:
                skipped += 1
                continue
            docs.append({"doc_index": len(docs), "input_ids": tokens})
            if len(docs) >= n_docs:
                break

    if not docs:
        print(f"ERROR: No documents with >= {min_tokens} tokens found.")
        sys.exit(1)
    print(f"Selected {len(docs)} doc(s) with >= {min_tokens} tokens (offset={doc_offset}).\n")
    return docs


# ---------------------------------------------------------------------------
# Prepare doc ladders
# ---------------------------------------------------------------------------

def prepare_doc_ladders(docs, context_lengths, score_window):
    prepared = []
    for doc in docs:
        L = len(doc["input_ids"])
        valid = [c for c in context_lengths if c + score_window <= L]
        if not valid:
            continue
        c_max = max(valid)
        ladder = sorted(c for c in context_lengths if c <= c_max)
        prepared.append({
            "doc_index": doc["doc_index"],
            "doc_len": L,
            "input_ids": doc["input_ids"],
            "c_max": c_max,
            "P": c_max,
            "ladder": ladder,
        })
    return prepared


# ---------------------------------------------------------------------------
# Extract per-token losses from vLLM prompt_logprobs output
# ---------------------------------------------------------------------------

def extract_losses(output, token_ids, context_len, score_window):
    """
    Extract per-token negative log-probs for the scoring window from
    vLLM's prompt_logprobs output.

    Args:
        output: vLLM RequestOutput with prompt_logprobs
        token_ids: the full input token sequence for this request
        context_len: c — the scoring window starts at position c
        score_window: W — number of tokens to score

    Returns:
        numpy array of shape (W,) with per-token losses (-logprob)
    """
    plogprobs = output.prompt_logprobs
    losses = np.zeros(score_window, dtype=np.float32)

    for j in range(score_window):
        pos = context_len + j
        if plogprobs[pos] is not None:
            token_id = token_ids[pos]
            if token_id in plogprobs[pos]:
                losses[j] = -plogprobs[pos][token_id].logprob
            else:
                # Fallback: take whatever entry is there
                entry = next(iter(plogprobs[pos].values()))
                losses[j] = -entry.logprob
        else:
            losses[j] = 0.0
    return losses


# ---------------------------------------------------------------------------
# Score all docs using vLLM
# ---------------------------------------------------------------------------

def score_all_docs(prepared, context_lengths, score_window, model_name, tp, gpu_mem):
    """
    Score all docs at all context lengths using vLLM.

    Strategy: for each context length, submit ALL eligible docs as a batch.
    vLLM handles scheduling internally (max_num_seqs controls concurrency).
    With chunked prefill, vLLM processes long prompts in manageable chunks.
    """
    from vllm import LLM, SamplingParams

    max_ctx = max(context_lengths)
    max_seq = max_ctx + score_window

    print(f"Initializing vLLM (tp={tp}, max_model_len={max_seq}, gpu_mem={gpu_mem})...")
    t0 = time.time()
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        max_model_len=max_seq,
        max_num_seqs=4,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
    )
    print(f"vLLM initialized in {time.time() - t0:.1f}s\n")

    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=0,
        temperature=0,
    )

    doc_losses = {d["doc_index"]: {} for d in prepared}
    timing_log = []

    for c in sorted(context_lengths):
        eligible = [d for d in prepared if c in d["ladder"]]
        if not eligible:
            continue

        seq_len = c + score_window

        # Build prompts for all eligible docs at this context length
        prompts = []
        prompt_token_lists = []
        doc_indices = []
        for d in eligible:
            start = d["P"] - c
            end = d["P"] + score_window
            token_ids = d["input_ids"][start:end]
            prompts.append({"prompt_token_ids": token_ids})
            prompt_token_lists.append(token_ids)
            doc_indices.append(d["doc_index"])

        print(f"  ctx={c//1024}K  seq_len={seq_len:>7}  n_docs={len(eligible)}", end="  ", flush=True)

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0

        # Extract losses
        for out, token_ids, doc_idx in zip(outputs, prompt_token_lists, doc_indices):
            doc_losses[doc_idx][c] = extract_losses(out, token_ids, c, score_window)

        per_doc = elapsed / len(eligible)
        total_tokens = len(eligible) * seq_len
        tps = total_tokens / elapsed
        print(f"time={elapsed:.2f}s  per_doc={per_doc:.3f}s  ({tps:.0f} tok/s)")

        timing_log.append({
            "context": c,
            "n_docs": len(eligible),
            "total_time_s": round(elapsed, 3),
            "per_doc_s": round(per_doc, 4),
            "tokens_per_sec": round(tps),
        })

    return doc_losses, timing_log


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pairwise_metrics(losses, alpha, beta):
    results = {}
    context_lengths = sorted(losses.keys())
    for i, c_short in enumerate(context_lengths):
        for c_long in context_lengths[i + 1:]:
            cb = losses[c_short] - losses[c_long]
            pair_key = f"{c_short}v{c_long}"
            is_key = (cb > alpha) & (losses[c_long] < (-1 * beta))
            results[pair_key] = {
                "ktr": round(float(np.mean(is_key)), 6),
                "mcb": round(float(np.mean(cb)), 6),
                "frac_pos": round(float(np.mean(cb > 0)), 6),
                "cb_p90": round(float(np.percentile(cb, 90)), 6),
                "n_scored": int(len(cb)),
            }
    return results


def aggregate_results(results):
    all_pairs = set()
    for r in results:
        all_pairs.update(r["pairwise"].keys())
    agg = {}
    for pair_key in sorted(all_pairs):
        vals = [r["pairwise"][pair_key] for r in results if pair_key in r["pairwise"]]
        if not vals:
            continue
        agg[pair_key] = {
            "n_docs": len(vals),
            "ktr_mean": round(float(np.mean([v["ktr"] for v in vals])), 6),
            "mcb_mean": round(float(np.mean([v["mcb"] for v in vals])), 6),
            "frac_pos_mean": round(float(np.mean([v["frac_pos"] for v in vals])), 6),
            "cb_p90_mean": round(float(np.mean([v["cb_p90"] for v in vals])), 6),
        }
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Context ladder scoring (vLLM)")
    parser.add_argument("--gcs-path", type=str, required=True)
    parser.add_argument("--token-column", type=str, default="tokens")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--n-docs", type=int, default=10)
    parser.add_argument("--min-tokens", type=int, default=14336)
    parser.add_argument("--score-window", type=int, default=10240)
    parser.add_argument("--context-lengths", type=str, default="4096,8192,16384,32768,65536,131072")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=-2.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--doc-offset", type=int, default=0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-mem", type=float, default=0.50)
    args = parser.parse_args()

    # Accept semicolons for CSV compatibility
    ctx_str = args.context_lengths.replace(";", ",")
    context_lengths = sorted(int(x) for x in ctx_str.split(","))

    # GPU info
    print("=" * 60)
    try:
        result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                                capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().split("\n"):
            if "GPU[0]" in line:
                print(line.strip())
    except Exception as e:
        print(f"Could not query GPU: {e}")
    print("=" * 60 + "\n")

    docs = load_docs_from_gcs(args.gcs_path, args.n_docs, args.min_tokens,
                               args.token_column, doc_offset=args.doc_offset)
    prepared = prepare_doc_ladders(docs, context_lengths, args.score_window)
    if not prepared:
        print("No documents eligible for scoring.")
        return
    print(f"Prepared {len(prepared)} docs for scoring.\n")

    total_t0 = time.time()
    doc_losses, timing_log = score_all_docs(
        prepared, context_lengths, args.score_window,
        args.model, args.tp, args.gpu_mem,
    )
    total_elapsed = time.time() - total_t0
    print(f"\nTotal scoring time: {total_elapsed:.1f}s for {len(prepared)} docs "
          f"({total_elapsed / len(prepared):.2f}s/doc avg)\n")

    results = []
    for d in prepared:
        losses = doc_losses[d["doc_index"]]
        if not losses:
            continue
        pairwise = compute_pairwise_metrics(losses, args.alpha, args.beta)
        results.append({
            "doc_index": d["doc_index"],
            "doc_len": d["doc_len"],
            "c_max": d["c_max"],
            "score_window_start": d["P"],
            "score_window_size": args.score_window,
            "context_lengths_used": d["ladder"],
            "pairwise": pairwise,
        })

    # Print summary
    print(f"{'doc':>5}  {'len':>8}  {'c_max':>7}  {'ladder':>25}")
    print("-" * 55)
    for row in results:
        ladder_str = ",".join(f"{c // 1024}K" for c in row["context_lengths_used"])
        print(f"{row['doc_index']:>5}  {row['doc_len']:>8}  {row['c_max']:>7}  {ladder_str:>25}")
        for pk in sorted(row["pairwise"].keys()):
            p = row["pairwise"][pk]
            print(f"        {pk}: KTR={p['ktr']:.4f}  MCB={p['mcb']:.3f}  "
                  f"FracPos={p['frac_pos']:.3f}  P90={p['cb_p90']:.3f}")

    if not results:
        print("\nNo documents scored.")
        return

    agg = aggregate_results(results)
    print(f"\n{'='*60}")
    print(f"AGGREGATE ({len(results)} docs)")
    print(f"{'='*60}")
    for pk, v in sorted(agg.items()):
        print(f"  {pk} (n={v['n_docs']}): KTR={v['ktr_mean']:.4f}  MCB={v['mcb_mean']:.3f}  "
              f"FracPos={v['frac_pos_mean']:.3f}  P90={v['cb_p90_mean']:.3f}")

    if args.output:
        output_data = {
            "config": {
                "model": args.model, "gcs_path": args.gcs_path,
                "score_window": args.score_window, "context_lengths": context_lengths,
                "alpha": args.alpha, "beta": args.beta,
                "n_docs": args.n_docs, "doc_offset": args.doc_offset,
                "tp": args.tp, "backend": "vllm",
            },
            "timing": {
                "total_s": round(total_elapsed, 1),
                "per_doc_avg_s": round(total_elapsed / len(prepared), 2),
                "per_context": timing_log,
            },
            "results": results,
            "aggregate": agg,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
