"""
Sample docs from KTR deciles, fetch text excerpts, and use the Claude API
to analyze what linguistic and structural features drive context dependency.

Loads all ladder score parquets, assigns 10 deciles on ktr_4096v32768,
samples N docs from target deciles (D1/D5/D9/D10 by default), fetches
beginning and scoring-window text via the tokenized parquets, then calls
Claude to characterize each decile and identify long-range dependency spans.
Writes a markdown report.

Usage:
    python scripts/report_ktr_deciles.py \\
        --gcs-scores-path gs://consus-dataproc/ocr/ia-ascm/scores/context_ladder_32k \\
        --gcs-text-path gs://consus-dataproc/ocr/ia-ascm/text/tokenized_fulldocs \\
        --output /tmp/ktr_decile_report.md
"""

import argparse
import io
import json
import os
import subprocess
import sys

import anthropic
import pandas as pd

SCORE_COL = "ktr_4096v32768"
N_DECILES = 10
DEFAULT_TARGET_DECILES = "1,5,9,10"  # 1-indexed display labels


# ---------------------------------------------------------------------------
# GCS helpers
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
        raise RuntimeError(
            f"gsutil cat failed for {gcs_path}: {result.stderr.decode().strip()}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Load scores and assign deciles
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
    print(
        f"Docs with valid {SCORE_COL}: {len(valid):,}  "
        f"(excluded {len(df) - len(valid):,} NaN/too-short)",
        flush=True,
    )
    for d in range(N_DECILES):
        subset = valid[valid["_decile"] == d]
        print(
            f"  D{d+1}: {len(subset):,} docs  "
            f"ktr=[{subset[SCORE_COL].min():.4f}, {subset[SCORE_COL].max():.4f}]"
        )
    return valid


# ---------------------------------------------------------------------------
# Build text index
# ---------------------------------------------------------------------------

def build_text_index(gcs_text_path):
    lines = gcs_list(gcs_text_path, recursive=True)
    index = {os.path.basename(p): p for p in lines if p.endswith(".parquet")}
    if not index:
        # Fallback: non-recursive listing (handles flat single-file datasets)
        lines = gcs_list(gcs_text_path, recursive=False)
        index = {os.path.basename(p): p for p in lines if p.endswith(".parquet")}
    if not index:
        print(f"ERROR: no parquet files under {gcs_text_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Indexed {len(index)} tokenized parquet file(s).", flush=True)
    return index


# ---------------------------------------------------------------------------
# Fetch and decode text excerpts
# ---------------------------------------------------------------------------

def fetch_excerpts(sampled, text_index, tokenizer, begin_tokens=1024, window_tokens=512):
    """
    Returns dict keyed by (source_file, row_index) ->
        {"begin": str, "window": str}

    Groups reads by source_file to minimise GCS round-trips.
    begin_tokens: tokens to decode from start of document
    window_tokens: tokens to decode from score_window_start
    """
    results = {}
    groups = sampled.groupby("source_file")
    n_groups = len(groups)

    for file_idx, (source_file, group) in enumerate(groups):
        if source_file not in text_index:
            print(
                f"  WARNING: '{source_file}' not in text index, "
                f"skipping {len(group)} doc(s)",
                file=sys.stderr,
            )
            continue

        gcs_path = text_index[source_file]
        print(f"  [{file_idx+1}/{n_groups}] {source_file} ({len(group)} doc(s))...", flush=True)

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
                print(f"  WARNING: row_index {idx} out of bounds in {source_file}", file=sys.stderr)
                continue

            if "tokens" not in parquet_df.columns:
                print(f"  WARNING: no 'tokens' column in {source_file}", file=sys.stderr)
                continue

            tokens = list(text_row["tokens"])
            win_start = int(row["score_window_start"])

            begin_text = tokenizer.decode(tokens[:begin_tokens], skip_special_tokens=True)
            window_text = tokenizer.decode(
                tokens[win_start : win_start + window_tokens], skip_special_tokens=True
            )
            results[(source_file, idx)] = {"begin": begin_text, "window": window_text}

    return results


# ---------------------------------------------------------------------------
# Claude API prompts and calls
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a researcher analyzing documents from {corpus_description} to understand what \
linguistic and structural features drive context-dependency in language models.

The KTR (Key Token Ratio) metric measures the fraction of tokens in a document's scoring window \
(roughly tokens 32,000–42,000 into the document) that require long context (32K tokens) rather \
than short context (4K tokens) to be predicted accurately. A token is "key" when the model's \
per-token loss is substantially lower with 32K context than with 4K context.

High KTR → many tokens in the scoring window benefit strongly from seeing text >4K tokens back.
Low KTR  → most tokens can be predicted well from local context alone.

Supporting metrics:
- MCB (Mean Cross-entropy Benefit): mean loss reduction on key tokens — measures the *magnitude* \
of the long-context benefit, not just how many tokens benefit.
- frac_pos: fraction of tokens with *any* improvement from long context (lower threshold than KTR).

You will be given {n_docs} document excerpts from a single KTR decile. For each document:
- Its KTR, MCB, and frac_pos scores
- A "beginning" excerpt: ~4K chars from the start of the document
- A "window" excerpt: ~2K chars from the scoring window region (tokens ~32K–42K in), \
which is the region where key tokens are actually measured

Your analysis must be intellectually rigorous, specific, and grounded in the text. \
Avoid vague generalisations. Quote directly from the excerpts where possible."""


def build_decile_prompt(decile_label, docs):
    ktr_vals = [d["ktr"] for d in docs]
    lines = [
        f"## Documents from Decile {decile_label}",
        f"KTR range in this decile: [{min(ktr_vals):.4f}, {max(ktr_vals):.4f}]",
        "",
    ]

    for i, doc in enumerate(docs, 1):
        lines += [
            f"### Document {i}  (source: {doc['source_file']}, row {doc['row_index']})",
            f"doc_len: {doc['doc_len']:,} tokens | "
            f"ktr_4096v32768: {doc['ktr']:.4f} | "
            f"mcb_4096v32768: {doc['mcb']:.4f} | "
            f"frac_pos_4096v32768: {doc['frac_pos']:.4f}",
            "",
            "**Beginning of document (~4K chars):**",
            "```",
            doc["begin"][:3000].strip(),
            "```",
            "",
            "**Scoring window region (~2K chars, from token ~32K onward):**",
            "```",
            doc["window"][:1500].strip(),
            "```",
            "",
        ]

    lines += [
        "---",
        "",
        "Provide a rigorous analysis of this decile covering the following four points. "
        "Be specific; use direct quotes and document numbers as evidence.",
        "",
        "**1. Document types and themes.**  "
        "What kinds of documents appear here? Identify dominant genres, subjects, and structural "
        "types. Are there patterns in how documents are organized (e.g. narrative, expository, "
        "reference, dialogue)?",
        "",
        "**2. Long-range dependency analysis.**  "
        "For 3–4 of these documents, identify and quote specific spans from the *window excerpt* "
        "that appear to constitute long-range dependencies. For each quoted span, explain: "
        "(a) what earlier information it depends on (visible in the beginning excerpt or "
        "reasonably inferred from the document type), (b) approximately how far back that "
        "information would be (in tokens/pages), and (c) why a 4K context window would "
        "likely fail to capture that dependency.",
        "",
        "**3. KTR explanation.**  "
        "Given the document types and dependency patterns you observed, why does this decile "
        "have the KTR level it does? What structural or semantic properties drive or suppress "
        "long-range context dependency in these documents?",
        "",
        "**4. Distinguishing features.**  "
        "What features most clearly mark this decile as different from the extremes of the "
        "distribution (very high or very low KTR)? What would you expect to change?",
    ]
    return "\n".join(lines)


def analyze_decile(client, decile_label, docs, claude_model, corpus_description):
    prompt = build_decile_prompt(decile_label, docs)
    print(f"  Calling Claude ({claude_model}) for {decile_label} (~{len(prompt):,} chars prompt)...", flush=True)
    response = client.messages.create(
        model=claude_model,
        max_tokens=4000,
        system=SYSTEM_PROMPT.format(n_docs=len(docs), corpus_description=corpus_description),
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    print(f"  Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out", flush=True)
    return response.content[0].text


SYNTHESIS_PROMPT = """\
You have analyzed four KTR deciles from {corpus_description}. \
The deciles were drawn from {n_total_docs:,} scored documents, with scoring based on \
comparing per-token losses at 4K vs 32K context length at roughly the 32K–42K token mark \
in each document.

The four decile analyses are reproduced below:

{analyses}

---

Write a concise synthesis (500–700 words) covering:

1. **Spectrum summary.** Describe the progression from D1 (lowest KTR) to D10 (highest KTR) \
in terms of document type, content structure, and linguistic properties. What is the single \
most predictive structural feature separating low- from high-KTR documents?

2. **Mechanisms of long-range dependency.** What are the primary mechanisms by which \
high-KTR documents create context dependency — coreference, callback to defined terms, \
narrative continuity, technical cross-references, something else? Are these mechanisms \
present in D5/D9 in weaker form, or qualitatively different?

3. **Data curation implications.** If you were selecting documents from this corpus to \
train a long-context language model, what would you prioritize or deprioritize based on \
KTR? Consider both the signal quality (do high-KTR documents provide more useful \
long-context training signal?) and any confounds (are there document types that have \
artificially high or low KTR for reasons unrelated to genuine semantic long-range dependency?).

Be specific and evidence-based. Reference observations from the individual decile analyses."""


def synthesize(client, analyses, claude_model, corpus_description, n_total_docs):
    analyses_block = "\n\n".join(
        f"### {label} Analysis\n\n{text}" for label, text in analyses.items()
    )
    prompt = SYNTHESIS_PROMPT.format(
        analyses=analyses_block, corpus_description=corpus_description,
        n_total_docs=n_total_docs,
    )
    print(f"  Calling Claude ({claude_model}) for synthesis (~{len(prompt):,} chars prompt)...", flush=True)
    response = client.messages.create(
        model=claude_model,
        max_tokens=3000,
        system=(
            "You are a researcher synthesizing findings about document context-dependency "
            "in language model training data. Be specific and evidence-based."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    print(f"  Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out", flush=True)
    return response.content[0].text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze KTR deciles via Claude and write a markdown report."
    )
    parser.add_argument("--gcs-scores-path", required=True,
                        help="GCS prefix with ladder score parquets")
    parser.add_argument("--gcs-text-path", required=True,
                        help="GCS prefix for tokenized parquets (recursively indexed)")
    parser.add_argument("--output", required=True,
                        help="Path to write the markdown report")
    parser.add_argument("--target-deciles", default=DEFAULT_TARGET_DECILES,
                        help="Comma-separated 1-indexed decile numbers (default: 1,5,9,10)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Docs to sample per target decile (default: 10)")
    parser.add_argument("--tokenizer-model", default="meta-llama/Meta-Llama-3.1-8B",
                        help="HF tokenizer model name")
    parser.add_argument("--claude-model", default="claude-sonnet-4-6",
                        help="Anthropic model to use for analysis (default: claude-sonnet-4-6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--excerpt-cache", default=None,
                        help="Path to JSONL cache of fetched excerpts. "
                             "If the file exists, skip GCS fetch and load from it. "
                             "If it does not exist, fetch and save to it.")
    parser.add_argument("--source-name", default="ia-ascm",
                        help="Short source name for report title (default: ia-ascm)")
    parser.add_argument("--corpus-description",
                        default="Internet Archive's scholarly and book corpus (ia-ascm)",
                        help="Corpus description used in Claude prompts")
    args = parser.parse_args()

    target_labels = [int(x) for x in args.target_deciles.split(",")]
    target_idx = [d - 1 for d in target_labels]  # convert to 0-indexed

    # When a cache exists, it contains all score metadata + excerpts — skip GCS entirely.
    if args.excerpt_cache and os.path.exists(args.excerpt_cache):
        print(f"\nLoading from cache: {args.excerpt_cache}", flush=True)
        cache_records = []
        with open(args.excerpt_cache) as f:
            for line in f:
                cache_records.append(json.loads(line))
        print(f"Loaded {len(cache_records)} cached records.", flush=True)

        # Reconstruct per-decile doc lists directly from cache
        docs_by_label = {}
        for rec in cache_records:
            label = rec["_target_label"]
            docs_by_label.setdefault(label, []).append(rec)
        for label in docs_by_label:
            docs_by_label[label].sort(key=lambda r: r["ktr"])

        client = anthropic.Anthropic()
        analyses = {}
        for d_label in target_labels:
            label = f"D{d_label}"
            docs = docs_by_label.get(label, [])
            if not docs:
                print(f"  WARNING: no cached docs for {label}, skipping", file=sys.stderr)
                continue
            print(f"\nAnalyzing {label} ({len(docs)} docs)...", flush=True)
            analyses[label] = analyze_decile(client, label, docs, args.claude_model, args.corpus_description)
        n_total_docs = sum(len(v) for v in docs_by_label.values())

    else:
        # 1. Load scores and assign deciles
        df = load_scores(args.gcs_scores_path)
        df = assign_deciles(df)
        n_total_docs = len(df)

        # 2. Sample target deciles
        sampled_parts = []
        for d_idx, d_label in zip(target_idx, target_labels):
            subset = df[df["_decile"] == d_idx]
            n = min(args.n_samples, len(subset))
            sample = subset.sample(n=n, random_state=args.seed).copy()
            sample["_target_label"] = f"D{d_label}"
            sampled_parts.append(sample)
        sampled = pd.concat(sampled_parts, ignore_index=True)
        print(f"\nSampled {len(sampled)} docs total.", flush=True)

        # 3. Build text index + load tokenizer
        print("\nBuilding text index...", flush=True)
        text_index = build_text_index(args.gcs_text_path)

        print(f"Loading tokenizer: {args.tokenizer_model}...", flush=True)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

        # 4. Fetch text excerpts
        print(f"\nFetching text excerpts for {len(sampled)} docs...", flush=True)
        excerpts = fetch_excerpts(sampled, text_index, tokenizer)
        print(f"Fetched {len(excerpts)} excerpts.", flush=True)

        # 5. Call Claude per decile
        client = anthropic.Anthropic()
        analyses = {}

        docs_by_label = {}
        for d_idx, d_label in zip(target_idx, target_labels):
            label = f"D{d_label}"
            group = sampled[sampled["_target_label"] == label].sort_values(SCORE_COL)
            docs = []
            for _, row in group.iterrows():
                key = (row["source_file"], int(row["row_index"]))
                if key not in excerpts:
                    continue
                docs.append({
                    "_target_label": label,
                    "source_file": row["source_file"],
                    "row_index": int(row["row_index"]),
                    "doc_len": int(row["doc_len"]),
                    "ktr": float(row[SCORE_COL]),
                    "mcb": float(row.get("mcb_4096v32768", float("nan"))),
                    "frac_pos": float(row.get("frac_pos_4096v32768", float("nan"))),
                    **excerpts[key],
                })
            docs_by_label[label] = docs

        if args.excerpt_cache:
            with open(args.excerpt_cache, "w") as f:
                for docs in docs_by_label.values():
                    for rec in docs:
                        f.write(json.dumps(rec) + "\n")
            print(f"Cache written to {args.excerpt_cache}", flush=True)

        for d_label in target_labels:
            label = f"D{d_label}"
            docs = docs_by_label.get(label, [])
            if not docs:
                print(f"  WARNING: no docs fetched for {label}, skipping", file=sys.stderr)
                continue
            print(f"\nAnalyzing {label} ({len(docs)} docs)...", flush=True)
            analyses[label] = analyze_decile(client, label, docs, args.claude_model, args.corpus_description)

    # 6. Synthesis
    synthesis = None
    if len(analyses) >= 2:
        print("\nGenerating synthesis...", flush=True)
        synthesis = synthesize(client, analyses, args.claude_model, args.corpus_description, n_total_docs)

    # 7. Write report
    # Build decile stats from whichever docs we have
    decile_stats = {}
    for d_label in target_labels:
        label = f"D{d_label}"
        if args.excerpt_cache and os.path.exists(args.excerpt_cache):
            if "cache_records" not in dir():
                cache_records = []
                with open(args.excerpt_cache) as f:
                    for line in f:
                        cache_records.append(json.loads(line))
            recs = [r for r in cache_records if r["_target_label"] == label]
            if recs:
                ktrs = [r["ktr"] for r in recs]
                decile_stats[label] = (min(ktrs), max(ktrs), len(recs))
        elif "df" in dir():
            d_idx = d_label - 1
            subset = df[df["_decile"] == d_idx][SCORE_COL]
            if len(subset):
                decile_stats[label] = (subset.min(), subset.max(), len(subset))

    report_lines = [
        f"# KTR Decile Analysis: {args.source_name} Context Ladder (4K vs 32K)",
        "",
        "Analysis of what document features drive high vs low Key Token Ratio (KTR) "
        f"on `ktr_4096v32768`, based on {args.n_samples} sampled documents per decile.",
        "",
        "**Metric definitions:**",
        "- **KTR (Key Token Ratio)**: fraction of tokens in the scoring window "
        "(tokens ~32K–42K) where long-context (32K) loss is substantially lower than "
        "short-context (4K) loss",
        "- **MCB (Mean Cross-entropy Benefit)**: mean loss reduction on key tokens",
        "- **frac_pos**: fraction of tokens with *any* improvement from long context",
        "",
        f"**Decile ranges (all {n_total_docs:,} docs):**",
        "",
    ]

    for label, (kmin, kmax, n) in decile_stats.items():
        report_lines.append(f"- {label}: n={n:,}  ktr=[{kmin:.4f}, {kmax:.4f}]")

    report_lines += ["", "---", ""]

    for label, analysis in analyses.items():
        report_lines += [f"## {label}", "", analysis, "", "---", ""]

    if synthesis:
        report_lines += ["## Synthesis", "", synthesis, ""]

    report = "\n".join(report_lines)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
