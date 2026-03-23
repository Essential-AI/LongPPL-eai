"""
Identify and analyze actual key tokens in high-KTR (D10) documents.

Phase 1 (GPU): Sample D10 docs from context ladder scores, re-score them
with LLaMA at 4K and 32K context to get per-token losses, identify key tokens
(where long context substantially reduces loss), decode spans with context,
and cache results to JSONL.

Phase 2 (CPU): Load cache, send key token spans to Claude API for analysis
of why each span depends on long-range context, write markdown report.

Usage:
    # Phase 1 only (on GPU pod, uploads cache to GCS):
    python scripts/report_key_tokens.py \\
        --gcs-scores-path gs://consus-dataproc/ocr/arxiv/scores/context_ladder_32k_v2 \\
        --gcs-text-path gs://consus-dataproc/ocr/arxiv/text/tokenized_fulldocs_v2 \\
        --excerpt-cache /tmp/key_token_cache.jsonl \\
        --output-gcs gs://consus-dataproc/ocr/arxiv/scores/key_token_cache.jsonl \\
        --phase1-only

    # Phase 2 only (locally, from cached JSONL):
    python scripts/report_key_tokens.py \\
        --excerpt-cache ~/kurt/analysis/arxiv_v2_ktr_decile/key_token_cache.jsonl \\
        --output ~/kurt/analysis/arxiv_v2_ktr_decile/key_token_report.md \\
        --source-name arxiv-v2 \\
        --corpus-description "arXiv scholarly papers"
"""

import argparse
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

SCORE_COL = "ktr_4096v32768"
N_DECILES = 10


# ---------------------------------------------------------------------------
# GCS helpers (from report_ktr_deciles.py)
# ---------------------------------------------------------------------------

def gcs_list(gcs_prefix, recursive=False):
    cmd = ["gsutil", "ls"]
    if recursive:
        cmd.append("-r")
    cmd.append(gcs_prefix.rstrip("/") + "/")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR listing {gcs_prefix}: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return sorted(l.strip() for l in result.stdout.strip().split("\n") if l.strip())


def gcs_cat(gcs_path):
    result = subprocess.run(["gsutil", "cat", gcs_path], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"gsutil cat failed for {gcs_path}: {result.stderr.decode().strip()}")
    return result.stdout


def gcs_upload(local_path, gcs_path):
    result = subprocess.run(["gsutil", "cp", local_path, gcs_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: upload to {gcs_path} failed: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"Uploaded {local_path} -> {gcs_path}", flush=True)


# ---------------------------------------------------------------------------
# Score loading and decile assignment (from report_ktr_deciles.py)
# ---------------------------------------------------------------------------

def load_scores(gcs_scores_path):
    files = [l for l in gcs_list(gcs_scores_path) if l.endswith(".parquet")]
    if not files:
        print(f"ERROR: no parquet files under {gcs_scores_path}", file=sys.stderr)
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
            print(f"  {i+1}/{len(files)} files ({sum(len(d) for d in dfs):,} docs)", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total scored docs: {len(df):,}", flush=True)
    return df


def assign_deciles(df):
    valid = df[df[SCORE_COL].notna()].copy()
    valid["_decile"] = pd.qcut(valid[SCORE_COL], q=N_DECILES, labels=False, duplicates="drop")
    print(f"Docs with valid {SCORE_COL}: {len(valid):,}", flush=True)
    return valid


def build_text_index(gcs_text_path):
    lines = gcs_list(gcs_text_path, recursive=True)
    index = {os.path.basename(p): p for p in lines if p.endswith(".parquet")}
    if not index:
        lines = gcs_list(gcs_text_path, recursive=False)
        index = {os.path.basename(p): p for p in lines if p.endswith(".parquet")}
    if not index:
        print(f"ERROR: no parquet files under {gcs_text_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Indexed {len(index)} tokenized parquet file(s).", flush=True)
    return index


# ---------------------------------------------------------------------------
# Key token span extraction (adapted from smoke_key_ratio.py)
# ---------------------------------------------------------------------------

def extract_key_token_spans(input_ids_list, key_positions, tokenizer,
                            context_window=40):
    if not key_positions:
        return []

    max_pos = len(input_ids_list)
    sorted_positions = sorted(key_positions)

    # Group adjacent positions into spans
    spans = []
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
    for s_start, s_end in spans:
        key_ids = input_ids_list[s_start: s_end + 1]
        key_text = tokenizer.decode(key_ids, skip_special_tokens=True)

        ctx_start = max(0, s_start - context_window)
        ctx_end = min(max_pos, s_end + 1 + context_window)
        ctx_ids = input_ids_list[ctx_start: ctx_end]
        ctx_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)

        results.append({
            "span_start": s_start,
            "span_end": s_end,
            "n_key_tokens": s_end - s_start + 1,
            "key_text": key_text,
            "context_text": ctx_text,
        })
    return results


# ---------------------------------------------------------------------------
# Phase 1: GPU scoring
# ---------------------------------------------------------------------------

def run_phase1(args):
    import torch

    # Add scripts dir to path for imports
    sys.path.insert(0, os.path.dirname(__file__))
    from context_ladder import load_model, score_doc_at_context_lengths

    context_lengths = sorted(int(x) for x in args.context_lengths.split(","))

    # 1. Load scores, sample D10
    df = load_scores(args.gcs_scores_path)
    df = assign_deciles(df)
    d10 = df[df["_decile"] == N_DECILES - 1]
    n = min(args.n_samples, len(d10))
    sampled = d10.sample(n=n, random_state=args.seed)
    print(f"\nSampled {n} D10 docs (KTR range: "
          f"[{sampled[SCORE_COL].min():.4f}, {sampled[SCORE_COL].max():.4f}])\n", flush=True)

    # 2. Build text index
    print("Building text index...", flush=True)
    text_index = build_text_index(args.gcs_text_path)

    # 3. Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer_model}...", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    # 4. Load model
    model = load_model(args.model)

    # Hoist constants
    lm_head_weight = model.lm_head.weight
    loss_f = torch.nn.CrossEntropyLoss(reduction="none")

    # 5. Process each sampled doc
    records = []
    groups = sampled.groupby("source_file")

    for source_file, group in groups:
        if source_file not in text_index:
            print(f"  WARNING: '{source_file}' not in text index, skipping", file=sys.stderr)
            continue

        gcs_path = text_index[source_file]
        print(f"  Reading: {gcs_path}", flush=True)
        try:
            raw = gcs_cat(gcs_path)
            parquet_df = pd.read_parquet(io.BytesIO(raw))
        except RuntimeError as e:
            print(f"  WARNING: {e}, skipping", file=sys.stderr)
            continue

        for _, row in group.iterrows():
            idx = int(row["row_index"])
            try:
                text_row = parquet_df.iloc[idx]
            except IndexError:
                print(f"  WARNING: row_index {idx} out of bounds", file=sys.stderr)
                continue

            tokens = list(text_row["tokens"])
            doc_len = len(tokens)
            max_needed = max(context_lengths) + args.score_window

            input_ids = torch.tensor(tokens[:max_needed], dtype=torch.long, device=model.device)

            print(f"  Scoring doc {source_file}:{idx} ({doc_len} tokens)...", end=" ", flush=True)

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

            c_short = min(context_lengths)
            c_long = max(context_lengths)

            if c_short not in losses or c_long not in losses:
                print(f"missing context lengths, skipping")
                continue

            # Identify key tokens
            cb = losses[c_short] - losses[c_long]
            is_key = (cb > args.alpha) & (losses[c_long] < -args.beta)
            key_indices = np.where(is_key)[0]
            key_positions = [P + int(i) for i in key_indices]

            n_key = len(key_positions)
            print(f"{n_key} key tokens ({n_key/len(cb)*100:.1f}%)")

            # Extract spans with loss stats
            spans = extract_key_token_spans(
                tokens, key_positions, tokenizer,
                context_window=args.span_context,
            )

            # Augment spans with loss statistics
            for span in spans:
                span_indices = [span["span_start"] + j - P
                                for j in range(span["span_start"], span["span_end"] + 1)
                                if 0 <= span["span_start"] + j - P < len(cb)]
                # Actually: indices into cb array
                span_cb_indices = [pos - P for pos in range(span["span_start"], span["span_end"] + 1)
                                   if 0 <= pos - P < len(cb)]
                if span_cb_indices:
                    span_cb = cb[span_cb_indices]
                    span["mean_cb"] = round(float(np.mean(span_cb)), 3)
                    span["max_cb"] = round(float(np.max(span_cb)), 3)
                else:
                    span["mean_cb"] = 0.0
                    span["max_cb"] = 0.0

            # Sort by mean_cb descending, limit
            spans.sort(key=lambda s: s["mean_cb"], reverse=True)
            spans = spans[:args.max_spans_per_doc]

            # Decode beginning of document
            begin_text = tokenizer.decode(tokens[:1024], skip_special_tokens=True)

            record = {
                "source_file": source_file,
                "row_index": idx,
                "doc_len": doc_len,
                "ktr": float(row[SCORE_COL]),
                "mcb": float(row.get("mcb_4096v32768", float("nan"))),
                "frac_pos": float(row.get("frac_pos_4096v32768", float("nan"))),
                "score_window_start": int(P),
                "n_key_tokens": n_key,
                "n_total_scored": len(cb),
                "begin_text": begin_text,
                "key_spans": spans,
            }
            records.append(record)

    # 6. Write cache
    cache_path = args.excerpt_cache or "/tmp/key_token_cache.jsonl"
    with open(cache_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nCache written: {cache_path} ({len(records)} docs)", flush=True)

    if args.output_gcs:
        gcs_upload(cache_path, args.output_gcs)

    return records


# ---------------------------------------------------------------------------
# Phase 2: Claude analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a researcher analyzing key tokens in documents from {corpus_description} to understand \
what specific linguistic and structural features create long-range context dependency in \
language models.

A "key token" is a token in the document's scoring window (roughly tokens 32,000–42,000 into \
the document) where the model's per-token loss is substantially lower with 32K context than \
with 4K context. Specifically, the loss difference (context benefit) exceeds {alpha} nats and \
the 32K-context loss is below {neg_beta} nats.

You will be given a document with its beginning excerpt (to establish what the document is about) \
and the actual key token spans identified by the model, each with surrounding context and the \
measured context benefit in nats. Your job is to explain *why* each key token span requires \
long-range context — what specific information from earlier in the document is needed to \
predict these tokens accurately.

Be specific. Quote from the text. Name the mechanism (coreference, cross-reference to a named \
equation/theorem, callback to a defined term, narrative continuity, etc.)."""


def build_doc_prompt(doc):
    lines = [
        f"## Document: {doc['source_file']}, row {doc['row_index']}",
        f"doc_len: {doc['doc_len']:,} tokens | "
        f"KTR: {doc['ktr']:.4f} | MCB: {doc['mcb']:.4f} | frac_pos: {doc['frac_pos']:.4f}",
        f"Score window starts at token {doc['score_window_start']:,} | "
        f"{doc['n_key_tokens']} key tokens out of {doc['n_total_scored']:,} scored "
        f"({doc['n_key_tokens']/doc['n_total_scored']*100:.1f}%)",
        "",
        "**Beginning of document (~4K chars):**",
        "```",
        doc["begin_text"][:3000].strip(),
        "```",
        "",
        f"**Key token spans ({len(doc['key_spans'])} spans, sorted by context benefit):**",
        "",
    ]

    for i, span in enumerate(doc["key_spans"], 1):
        lines += [
            f"### Span {i} (tokens {span['span_start']:,}–{span['span_end']:,}, "
            f"{span['n_key_tokens']} key token(s), "
            f"mean CB: {span['mean_cb']:.2f} nats, max CB: {span['max_cb']:.2f} nats)",
            "",
            f"Key text: `{span['key_text']}`",
            "",
            f"Context: {span['context_text']}",
            "",
        ]

    lines += [
        "---",
        "",
        "For each key token span above, explain:",
        "1. **What earlier information does this span depend on?** "
        "Be specific — name the entity, equation, definition, or concept from earlier "
        "in the document that is needed.",
        "2. **What is the mechanism of dependency?** "
        "(coreference to a named entity, cross-reference to a numbered equation/theorem, "
        "callback to a technical term defined earlier, structural pattern continuation, etc.)",
        "3. **How far back is the needed information?** "
        "Estimate in tokens or pages, based on the document structure visible in the beginning excerpt.",
        "4. **Why can't 4K context capture this?** "
        "Explain why the relevant information is beyond the 4K-token horizon at the scoring window position.",
        "",
        "Then provide a brief overall summary of the dominant dependency mechanisms in this document.",
    ]
    return "\n".join(lines)


SYNTHESIS_PROMPT = """\
You have analyzed key token spans from {n_docs} high-KTR documents from {corpus_description}. \
The documents are from the top 10% of the KTR distribution (D10), meaning they have the \
strongest long-range context dependency.

The per-document analyses are:

{analyses}

---

Write a concise synthesis (400–600 words) covering:

1. **Taxonomy of dependency mechanisms.** What are the distinct mechanisms by which these \
tokens depend on long-range context? Provide a ranked list with examples.

2. **Token-level patterns.** Do key tokens tend to be specific types (variable names, citation \
markers, theorem numbers, etc.)? What token types are over-represented among key tokens?

3. **Clustering.** Do key tokens cluster in specific regions of the scoring window, or are they \
distributed? Do they tend to appear in specific document structures (proofs, equations, \
references sections)?

4. **Training implications.** What does this tell us about what a long-context model needs to \
learn from these documents? What specific capabilities are being tested by these key tokens?

Be specific and evidence-based."""


def analyze_doc(client, doc, claude_model, corpus_description, alpha, beta):
    prompt = build_doc_prompt(doc)
    print(f"  Calling Claude for {doc['source_file']}:{doc['row_index']} "
          f"(~{len(prompt):,} chars)...", flush=True)
    response = client.messages.create(
        model=claude_model,
        max_tokens=5000,
        system=SYSTEM_PROMPT.format(
            corpus_description=corpus_description,
            alpha=alpha, neg_beta=-beta,
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    print(f"  Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out", flush=True)
    return response.content[0].text


def synthesize(client, analyses, claude_model, corpus_description):
    analyses_block = "\n\n".join(
        f"### Document {i+1}\n\n{text}" for i, text in enumerate(analyses)
    )
    prompt = SYNTHESIS_PROMPT.format(
        analyses=analyses_block,
        corpus_description=corpus_description,
        n_docs=len(analyses),
    )
    print(f"  Calling Claude for synthesis (~{len(prompt):,} chars)...", flush=True)
    response = client.messages.create(
        model=claude_model,
        max_tokens=3000,
        system="You are a researcher synthesizing findings about key token dependency "
               "mechanisms in long-context language model training data.",
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    print(f"  Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out", flush=True)
    return response.content[0].text


def run_phase2(records, args):
    import anthropic
    client = anthropic.Anthropic()

    analyses = []
    for rec in records:
        if not rec["key_spans"]:
            print(f"  Skipping {rec['source_file']}:{rec['row_index']} (no key spans)")
            continue
        text = analyze_doc(
            client, rec, args.claude_model, args.corpus_description,
            args.alpha, args.beta,
        )
        analyses.append(text)

    synthesis = None
    if len(analyses) >= 2:
        print("\nGenerating synthesis...", flush=True)
        synthesis = synthesize(client, analyses, args.claude_model, args.corpus_description)

    # Build report
    report_lines = [
        f"# Key Token Analysis: D10 Documents from {args.source_name}",
        "",
        "Identification and analysis of actual key tokens in high-KTR documents. "
        "Key tokens are positions in the scoring window (tokens ~32K–42K) where the model's "
        f"loss drops by >{args.alpha} nats when given 32K context instead of 4K.",
        "",
        f"**Documents analyzed:** {len(records)}",
        "",
        "---",
        "",
    ]

    for i, (rec, analysis) in enumerate(zip(records, analyses)):
        # Extract title from begin_text
        title = ""
        for line in rec["begin_text"].split("\n"):
            line = line.strip().lstrip("#").strip()
            if line and len(line) > 10:
                title = line[:120]
                break

        report_lines += [
            f"## Document {i+1}: {title}",
            "",
            f"**Source:** `{rec['source_file']}` row {rec['row_index']}  ",
            f"**doc_len:** {rec['doc_len']:,} tokens | "
            f"**KTR:** {rec['ktr']:.4f} | **MCB:** {rec['mcb']:.4f}  ",
            f"**Key tokens:** {rec['n_key_tokens']} / {rec['n_total_scored']:,} scored "
            f"({rec['n_key_tokens']/rec['n_total_scored']*100:.1f}%)",
            "",
            "### Key Token Spans",
            "",
            "| # | Position | Key Tokens | Key Text | Mean CB | Max CB |",
            "|---|----------|-----------|----------|---------|--------|",
        ]

        for j, span in enumerate(rec["key_spans"], 1):
            key_text_short = span["key_text"][:50]
            key_text_short = key_text_short.replace("\\", "\\\\")
            key_text_short = key_text_short.replace("|", "\\|")
            key_text_short = key_text_short.replace("`", "\\`")
            key_text_short = key_text_short.replace("*", "\\*")
            key_text_short = key_text_short.replace("\n", "⏎")
            report_lines.append(
                f"| {j} | {span['span_start']:,}–{span['span_end']:,} | "
                f"{span['n_key_tokens']} | `{key_text_short}` | "
                f"{span['mean_cb']:.2f} | {span['max_cb']:.2f} |"
            )

        report_lines += ["", "### Analysis", "", analysis, "", "---", ""]

    if synthesis:
        report_lines += ["## Synthesis", "", synthesis, ""]

    report = "\n".join(report_lines)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport written to {args.output}")
    else:
        print(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Key token analysis for D10 documents")
    parser.add_argument("--gcs-scores-path", default=None,
                        help="GCS prefix with context ladder score parquets")
    parser.add_argument("--gcs-text-path", default=None,
                        help="GCS prefix for tokenized parquets")
    parser.add_argument("--output", default=None, help="Path for markdown report")
    parser.add_argument("--output-gcs", default=None,
                        help="GCS path to upload cache JSONL (Phase 1)")
    parser.add_argument("--excerpt-cache", default=None,
                        help="JSONL cache path. If exists, skip Phase 1.")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run Phase 1 (GPU scoring) only, skip Claude analysis")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--tokenizer-model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--score-window", type=int, default=10240)
    parser.add_argument("--context-lengths", default="4096,32768")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=-2.0)
    parser.add_argument("--span-context", type=int, default=40)
    parser.add_argument("--max-spans-per-doc", type=int, default=20)
    parser.add_argument("--claude-model", default="claude-sonnet-4-6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-name", default="arxiv-v2")
    parser.add_argument("--corpus-description", default="arXiv scholarly papers")
    args = parser.parse_args()

    # Phase 1: GPU scoring (skip if cache exists)
    if args.excerpt_cache and os.path.exists(args.excerpt_cache):
        print(f"\nLoading from cache: {args.excerpt_cache}", flush=True)
        records = []
        with open(args.excerpt_cache) as f:
            for line in f:
                records.append(json.loads(line))
        print(f"Loaded {len(records)} cached docs.", flush=True)
    else:
        if not args.gcs_scores_path or not args.gcs_text_path:
            print("ERROR: --gcs-scores-path and --gcs-text-path required for Phase 1",
                  file=sys.stderr)
            sys.exit(1)
        records = run_phase1(args)

    if args.phase1_only:
        print("Phase 1 complete. Exiting (--phase1-only).")
        return

    # Phase 2: Claude analysis
    if not records:
        print("No records to analyze.")
        return

    run_phase2(records, args)


if __name__ == "__main__":
    main()
