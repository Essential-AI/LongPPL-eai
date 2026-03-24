"""Minimal GPU scoring script for pre-sampled documents.

Reads a JSON file listing (source_file, row_index) pairs, fetches tokens from GCS,
scores each doc at 4K and 32K context, extracts key token spans, and writes a
key_token_cache.jsonl compatible with report_key_tokens.py Phase 2.

Usage:
    python score_sampled_docs.py \
        --samples /tmp/samples.json \
        --gcs-text-path gs://consus-dataproc/ocr/ia-ascm/text/tokenized_fulldocs \
        --output /tmp/key_token_cache.jsonl \
        --output-gcs gs://consus-dataproc/ocr/ia-ascm/scores/key_token_cache.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="JSON file with list of {source_file, row_index, doc_len, ktr}")
    parser.add_argument("--gcs-text-path", required=True)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--tokenizer-model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=-2.0)
    parser.add_argument("--score-window", type=int, default=10240)
    parser.add_argument("--output", default="/tmp/key_token_cache.jsonl")
    parser.add_argument("--output-gcs", default=None)
    args = parser.parse_args()

    with open(args.samples) as f:
        samples = json.loads(f.read())
    print(f"Loaded {len(samples)} samples.", flush=True)

    # Build text index
    import gcsfs
    import pyarrow.parquet as pq
    fs = gcsfs.GCSFileSystem()

    all_files = sorted(fs.ls(args.gcs_text_path))
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    if not parquet_files:
        parquet_files = sorted(fs.glob(args.gcs_text_path.rstrip("/") + "/**/*.parquet"))
    if not parquet_files:
        # Fallback: non-recursive
        all_files = sorted(fs.ls(args.gcs_text_path))
        parquet_files = [f for f in all_files if f.endswith(".parquet")]
    text_index = {os.path.basename(p): p for p in parquet_files}
    print(f"Indexed {len(text_index)} text files.", flush=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    # Wait for GPU VRAM cleanup then load model
    print("Waiting 15s for GPU VRAM cleanup...", flush=True)
    time.sleep(15)

    sys.path.insert(0, os.path.dirname(__file__))
    from context_ladder import load_model, score_doc_at_context_lengths
    model = load_model(args.model)

    lm_head_weight = model.lm_head.weight
    loss_f = torch.nn.CrossEntropyLoss(reduction="none")
    context_lengths = [4096, 32768]

    records = []
    for sample in samples:
        sf = sample["source_file"]
        ri = sample["row_index"]
        print(f"\n  Reading: {text_index.get(sf, 'NOT FOUND')}", flush=True)

        if sf not in text_index:
            print(f"  WARNING: {sf} not in text index, skipping")
            continue

        with fs.open(text_index[sf], "rb") as fh:
            table = pq.read_table(fh, columns=["tokens"])
        tokens = table.column("tokens")[ri].as_py()
        doc_len = len(tokens)

        max_needed = max(context_lengths) + args.score_window
        input_ids = torch.tensor(tokens[:max_needed], dtype=torch.long, device=model.device)

        print(f"  Scoring doc {sf}:{ri} ({doc_len} tokens)...", end=" ", flush=True)
        try:
            losses, P, ladder = score_doc_at_context_lengths(
                input_ids, model, context_lengths, args.score_window,
                lm_head_weight, loss_f,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("OOM, skipping")
            continue

        del input_ids
        torch.cuda.empty_cache()

        if losses is None:
            print("too short, skipping")
            continue

        # Identify key tokens
        loss_short = losses[context_lengths[0]]
        loss_long = losses[context_lengths[1]]
        cb = loss_short - loss_long
        is_key = (cb > args.alpha) & (loss_long < -args.beta)
        key_indices = np.where(is_key)[0]
        n_key = int(is_key.sum())
        print(f"{n_key} key tokens ({100*n_key/args.score_window:.1f}%)", flush=True)

        # Extract key token spans
        key_spans = []
        if len(key_indices) > 0:
            # Group consecutive key tokens into spans
            groups = []
            start = key_indices[0]
            for i in range(1, len(key_indices)):
                if key_indices[i] - key_indices[i-1] > 5:  # gap > 5 tokens = new span
                    groups.append((start, key_indices[i-1]))
                    start = key_indices[i]
            groups.append((start, key_indices[-1]))

            # For each span, decode with context
            for gs, ge in groups:
                span_key_mask = is_key[gs:ge+1]
                span_cb = cb[gs:ge+1]
                ctx = 40  # context tokens around span
                abs_start = P + gs
                abs_end = P + ge
                decode_start = max(0, abs_start - ctx)
                decode_end = min(doc_len, abs_end + ctx + 1)
                context_text = tokenizer.decode(tokens[decode_start:decode_end], skip_special_tokens=True)
                key_text = tokenizer.decode(tokens[abs_start:abs_end+1], skip_special_tokens=True)

                key_spans.append({
                    "span_start": int(abs_start),
                    "span_end": int(abs_end),
                    "n_key_tokens": int(span_key_mask.sum()),
                    "key_text": key_text,
                    "context_text": context_text,
                    "mean_cb": round(float(span_cb[span_key_mask].mean()), 2),
                    "max_cb": round(float(span_cb[span_key_mask].max()), 2),
                })

            # Sort by mean_cb descending, keep top 20
            key_spans.sort(key=lambda s: s["mean_cb"], reverse=True)
            key_spans = key_spans[:20]

        # Decode beginning of document
        begin_text = tokenizer.decode(tokens[:1024], skip_special_tokens=True)

        records.append({
            "source_file": sf,
            "row_index": ri,
            "doc_len": doc_len,
            "ktr": sample["ktr"],
            "mcb": sample.get("mcb", 0),
            "frac_pos": sample.get("frac_pos", 0),
            "score_window_start": int(P),
            "n_key_tokens": n_key,
            "n_total_scored": args.score_window,
            "begin_text": begin_text,
            "key_spans": key_spans,
        })

    # Write cache
    with open(args.output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nCache written: {args.output} ({len(records)} docs)", flush=True)

    if args.output_gcs:
        subprocess.run(["gsutil", "cp", args.output, args.output_gcs], check=True)
        print(f"Uploaded {args.output} -> {args.output_gcs}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
