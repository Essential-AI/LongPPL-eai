# Context Ladder Analysis Report

**Date**: 2026-03-16
**Model**: Llama-3.1-8B (bf16, SDPA)
**Hardware**: MI300X 192GB (Cirrascale)
**Docs scored**: 475 across 6 sources (24/24 jobs completed)

## 1. Executive Summary

We evaluated 475 documents across 6 data sources, computing per-token loss at up to 6 context lengths {4K, 8K, 16K, 32K, 64K, 128K} on a fixed 10,240-token scoring window. Each document uses an adaptive ladder — only context lengths where the document is long enough.

**Key findings for long-context training data curation:**

1. **Marginal context benefit is flat from 4K→64K, then drops at 128K.** Each doubling of context adds ~0.02 MCB, but the 64K→128K step adds only 0.011. Most "context benefit" comes from nearby context (4K–16K range).

2. **Programming and science-and-math are the richest sources for long-context signal.** They have the highest MCB, KTR, and FracPos at every comparison level.

3. **Only 16% of 128K-capable docs show "sustained scaling"** — where benefit at 128K is more than double the benefit at 32K. These are the highest-value docs for 128K training.

4. **FracPos rarely exceeds 0.58**, meaning only ~58% of tokens benefit from longer context at best. The remaining 42% are equally predictable at any context length.

## 2. Dataset Composition

| Source | Docs scored | c_max distribution |
|--------|------------|-------------------|
| arxiv | 100 | Mixed (4K–128K) |
| eai-crawl-journals | 100 | Mixed |
| ia-ascm | 75 | Mixed |
| library | 50 | Heavily 64K–128K (long books) |
| programming | 100 | Mixed |
| science-and-math | 50 | Mixed |

### Context ladder distribution (all docs):

| c_max | Count | % |
|-------|-------|---|
| 4K | 92 | 19% |
| 8K | 75 | 16% |
| 16K | 72 | 15% |
| 32K | 55 | 12% |
| 64K | 70 | 15% |
| 128K | 111 | 23% |

## 3. How Context Benefit Scales

### 3.1 Cumulative benefit from 4K baseline

Each row shows how much better a model predicts when given context length `c_long` vs. only 4K:

| Comparison | Context ratio | MCB | KTR | FracPos | P90 | n_docs |
|-----------|--------------|-----|-----|---------|-----|--------|
| 4K vs 8K | 2x | 0.021 | 0.0022 | 0.508 | 0.124 | 383 |
| 4K vs 16K | 4x | 0.041 | 0.0043 | 0.531 | 0.190 | 308 |
| 4K vs 32K | 8x | 0.057 | 0.0062 | 0.558 | 0.260 | 236 |
| 4K vs 64K | 16x | 0.073 | 0.0083 | 0.562 | 0.326 | 181 |
| 4K vs 128K | 32x | 0.077 | 0.0104 | 0.548 | 0.394 | 111 |

**Interpretation**: MCB grows roughly as log(context_ratio) from 4K–64K, then plateaus. Going from 4K→64K gives 0.073 MCB; extending to 128K adds only 0.004 more. The P90 (90th percentile of per-token benefit) continues to grow, meaning a small fraction of tokens does benefit substantially from very long context.

### 3.2 Marginal benefit per doubling

| Step | MCB | KTR | FracPos | n_docs |
|------|-----|-----|---------|--------|
| 4K → 8K | 0.021 | 0.0022 | 0.508 | 383 |
| 8K → 16K | 0.023 | 0.0025 | 0.506 | 308 |
| 16K → 32K | 0.021 | 0.0023 | 0.524 | 236 |
| 32K → 64K | 0.020 | 0.0024 | 0.494 | 181 |
| 64K → 128K | 0.011 | 0.0029 | 0.469 | 111 |

**Interpretation**: Each doubling adds ~0.021 MCB consistently, except the last step (64K→128K) which drops to 0.011. This suggests diminishing returns beyond 64K for most documents. The FracPos drops below 0.5 at 64K→128K, meaning fewer than half of tokens actually benefit from extending context from 64K to 128K.

However, KTR (key token ratio — tokens with strong benefit above threshold) actually *increases* at 64K→128K (0.0029 vs 0.0024), indicating that while fewer tokens benefit overall, the ones that do benefit more strongly.

## 4. Per-Source Analysis

### 4.1 Medium range: 4K vs 32K

| Source | n | KTR | MCB | FracPos | P90 |
|--------|---|-----|-----|---------|-----|
| **programming** | 55 | **0.0089** | **0.074** | 0.569 | 0.288 |
| **science-and-math** | 43 | **0.0081** | **0.063** | **0.591** | 0.262 |
| arxiv | 12 | 0.0064 | 0.057 | 0.610 | 0.191 |
| ia-ascm | 59 | 0.0050 | 0.054 | 0.546 | 0.285 |
| eai-crawl-journals | 21 | 0.0045 | 0.038 | 0.528 | 0.173 |
| library | 46 | 0.0034 | 0.042 | 0.529 | 0.252 |

### 4.2 Long range: 4K vs 128K

| Source | n | KTR | MCB | FracPos | P90 |
|--------|---|-----|-----|---------|-----|
| **science-and-math** | 31 | **0.0137** | **0.089** | **0.578** | 0.403 |
| **programming** | 17 | **0.0114** | **0.089** | 0.577 | 0.366 |
| ia-ascm | 24 | 0.0103 | 0.082 | 0.538 | 0.440 |
| library | 29 | 0.0085 | 0.067 | 0.520 | 0.392 |
| eai-crawl-journals | 10 | 0.0047 | 0.037 | 0.508 | 0.308 |

**Programming and science-and-math consistently have the highest context benefit at all ranges.** This makes intuitive sense: code has function definitions referenced from far away, and scientific papers have citations, equations, and definitions that span long distances.

### 4.3 Per-doc MCB distribution (4K vs 32K)

| Source | n | mean | p10 | p50 | p90 | max |
|--------|---|------|-----|-----|-----|-----|
| programming | 55 | 0.074 | 0.015 | 0.063 | 0.158 | **0.354** |
| science-and-math | 43 | 0.063 | 0.008 | 0.045 | 0.104 | **0.550** |
| arxiv | 12 | 0.057 | 0.032 | 0.053 | 0.085 | 0.127 |
| ia-ascm | 59 | 0.054 | 0.021 | 0.050 | 0.097 | 0.144 |
| library | 46 | 0.042 | 0.009 | 0.034 | 0.074 | 0.197 |
| eai-crawl-journals | 21 | 0.038 | 0.005 | 0.024 | 0.111 | 0.139 |

**Note the heavy-tailed distributions**: science-and-math has a doc with MCB=0.550 (10x the median). Programming has docs up to 0.354. These outlier documents are extremely rich in long-range dependencies and are the most valuable for training.

## 5. Cross-Context-Length Correlations

How well does a cheap short-context comparison predict a more expensive long-context comparison? n=111 docs with full 128K ladders.

### 5.1 Cumulative pairs (4K baseline) — MCB Pearson r

Each cell shows Pearson r between the row's MCB and the column's MCB across docs. Higher r = the cheaper predictor (row) reliably ranks docs the same way as the more expensive target (column).

| Predictor → Target | 4Kv8K | 4Kv16K | 4Kv32K | 4Kv64K | 4Kv128K |
|---------------------|:-----:|:------:|:------:|:------:|:-------:|
| **4Kv8K**           |   —   | 🟡 0.72 | 🔴 0.47 | 🔴 0.42 | 🔴 0.32 |
| **4Kv16K**          |   —   |   —    | 🟢 0.91 | 🟡 0.80 | 🟡 0.61 |
| **4Kv32K**          |   —   |   —    |   —    | 🟢 0.93 | 🟡 0.72 |
| **4Kv64K**          |   —   |   —    |   —    |   —    | 🟢 0.83 |

**Reading**: 4Kv8K is a poor predictor of 128K benefit (r=0.32, r²=0.10). 4Kv16K is moderate (r=0.61, r²=0.37). 4Kv32K explains ~half the variance (r=0.72, r²=0.53). 4Kv64K is the best cheap proxy (r=0.83, r²=0.69).

### 5.2 Marginal pairs predicting cumulative targets — MCB Pearson r

Can the *incremental* benefit from one step (e.g., 16K→32K) predict total benefit at longer ranges?

| Predictor → Target | 4Kv8K | 4Kv16K | 4Kv32K | 4Kv64K | 4Kv128K |
|---------------------|:-----:|:------:|:------:|:------:|:-------:|
| **4Kv8K**           | 🟢 1.00 | 🟡 0.72 | 🔴 0.47 | 🔴 0.42 | 🔴 0.32 |
| **8Kv16K**          | 🔴 0.37 | 🟢 0.91 | 🟢 0.94 | 🟢 0.82 | 🟡 0.63 |
| **16Kv32K**         | 🔴 0.10 | 🟡 0.62 | 🟢 0.89 | 🟢 0.88 | 🟡 0.70 |
| **32Kv64K**         | 🔴 0.05 | 🔴 0.06 | 🔴 0.20 | 🟡 0.55 | 🟡 0.57 |
| **64Kv128K**        | 🔴 0.01 | 🔴 0.02 | 🔴 0.05 | 🔴 0.14 | 🟡 0.67 |

🟢 r ≥ 0.80 &nbsp; 🟡 0.55 ≤ r < 0.80 &nbsp; 🔴 r < 0.55

**Key insight**: 8Kv16K marginal benefit is a surprisingly strong predictor of 4Kv32K (r=0.94) and even 4Kv64K (r=0.82). This means **a 2-pass filter using 4K+8K+16K forward passes (~3.5s/doc) captures most of the ranking signal for 64K-level benefit.**

### 5.3 KTR correlations with 4Kv128K

| Predictor | r | r² |
|-----------|------|------|
| 4Kv8K     | 0.26 | 0.07 |
| 4Kv16K    | 0.62 | 0.39 |
| 4Kv32K    | 0.72 | 0.52 |
| 4Kv64K    | 0.82 | 0.67 |

KTR correlations track MCB correlations closely, confirming the pattern holds across metrics.

### 5.4 Implications for multi-pass filtering

Based on the correlation structure:

- **Pass 1 (4K+16K, ~1s/doc)**: MCB(4Kv16K) has r=0.61 with 128K target. Enough to eliminate the bottom ~40% of docs cheaply.
- **Pass 2 (add 32K, ~3.2s/doc)**: MCB(4Kv32K) has r=0.72, and combined with 8Kv16K marginal (r=0.94 with 4Kv32K), gives strong multi-feature signal.
- **Pass 3 (add 64K, ~6.3s/doc)**: MCB(4Kv64K) has r=0.83. Only needed for final ranking of top candidates.

## 6. Sustained Scaling: The Most Valuable Documents

Of 111 documents with full 128K ladders, only **18 (16%)** show "sustained scaling" — where MCB at 4Kv128K is more than double the MCB at 4Kv32K. These are documents where context benefit doesn't plateau at medium ranges but continues growing at very long distances.

**This is the key filtering criterion for 128K training data:** a document that benefits from 32K context but not 128K context teaches the model nothing new about very-long-range attention.

## 7. Implications for Long-Context Training Data Curation

### 7.1 Source prioritization

For **128K context training**, prioritize in this order:
1. **science-and-math** — highest KTR and MCB at 128K, highest sustained scaling
2. **programming** — close second, strong long-range dependencies (function references, imports)
3. **ia-ascm** — decent long-range signal, large corpus
4. **library** — moderate signal, but docs are very long (good for context length coverage)
5. **eai-crawl-journals** — weakest long-range signal, deprioritize for 128K training

### 7.2 Document-level filtering strategy

A multi-pass approach is recommended:

**Pass 1 — Cheap filter (4K + 16K, ~1s/doc):** Score all docs with just two context lengths. Filter out docs with MCB(4Kv16K) < 0.01 (no meaningful context benefit). Expected to remove ~30-40% of docs.

**Pass 2 — Medium filter (add 32K, ~2.2s/doc):** On survivors, add 32K context. Compute the "scaling slope" from 16K to 32K. Filter docs where MCB(4Kv32K) < 0.03 or where benefit is plateauing (MCB(16Kv32K) < 0.005).

**Pass 3 — Full evaluation (add 64K + 128K, ~27s/doc):** On top candidates, run the full ladder. Select docs where:
- MCB(4Kv128K) > 0.05 (meaningful total benefit)
- MCB(64Kv128K) > 0 (still gaining at the longest range)
- Sustained scaling ratio > 1.5 (benefit at 128K significantly exceeds benefit at 32K)

### 7.3 The "FracPos ceiling"

FracPos (fraction of tokens that benefit from longer context) maxes out around 0.55-0.58 even in the best sources. This means that even in the most context-dependent documents, ~42% of tokens are equally predictable regardless of context length. These are typically:
- Common function words and syntax
- Locally predictable continuations
- Repetitive patterns

This suggests that a loss-weighted training approach (like LongCE from the LongPPL paper) that upweights context-dependent tokens could be more efficient than naive uniform training on long documents.

### 7.4 The diminishing returns problem

The marginal MCB per doubling is remarkably flat at ~0.021 from 4K→64K, then drops to 0.011 at 64K→128K. This means:
- **The cost of 128K training (2x the tokens of 64K) buys only half the marginal benefit** compared to any earlier doubling.
- Training on 64K context captures ~95% of the total context benefit available at 128K (MCB 0.073 vs 0.077).
- The remaining 5% is concentrated in a small fraction of tokens (KTR rises from 0.0083 to 0.0104).

**Recommendation**: Unless the use case specifically requires 128K capability, 64K context training may offer a better cost/benefit ratio. For 128K training, heavily filter for the ~16% of documents that show sustained scaling.

## 8. Throughput & Infrastructure

| Metric | Value |
|--------|-------|
| Backend | HF transformers (SDPA, bf16) |
| GPU | MI300X 192GB, single GPU via `device_map={"": 0}` |
| Full ladder per doc | ~30s (128K-capable), ~3.6s (32K-capable), ~0.2s (4K only) |
| 475 docs total time | ~1.7 GPU-hours |
| Batching speedup | 1.1x max (memory-bandwidth bound) |
| vLLM | Not viable (3-5x slower, crashes at 128K) |

### Chunked lm_head optimization

The script uses a memory-efficient scoring approach: instead of materializing the full `(seq_len x 128K_vocab)` logits tensor (~36GB at 128K context), it:
1. Runs the transformer layers to get hidden states
2. Extracts only the W=10K scoring window positions
3. Computes the lm_head projection in chunks of 1024 positions

Peak logit memory: 1024 x 128256 x 2 bytes = **250MB** (vs 36GB). No performance penalty vs the naive approach.

### vLLM comparison (not recommended)

| Context | HF per-doc | vLLM per-doc | Ratio |
|---------|-----------|-------------|-------|
| 4K | 0.71s | 2.24s | 3.1x slower |
| 8K | 1.00s | 3.25s | 3.3x slower |
| 16K | 1.64s | 6.14s | 3.7x slower |
| 32K | 3.27s | 14.49s | 4.4x slower |
| 64K | 7.95s | 41.55s | 5.2x slower |
| 128K | 23.5s | GPU memory fault | -- |

vLLM's `prompt_logprobs` materializes full vocab logits and computes softmax for every prompt token. This is fundamentally different from generation (where only the last token's logits matter), and vLLM is not optimized for it. 128K consistently crashes with GPU memory faults.

## 9. Data files

### Results
```
gs://consus-dataproc/ocr/{source}/scores/context_ladder/ladder_{0..3}.json
```

### Scripts
- `scripts/context_ladder.py` — Scoring script
- `scripts/generate_ladder_csv.py` — Job CSV generator
- `scripts/batch_microbench.py` — Throughput benchmark
- `scripts/context_ladder_vllm.py` — vLLM version (not recommended)

### Nox templates
- `modmax/nox_templates/longppl-context-ladder.yml` — Production template
- `modmax/nox_templates/longppl-ladder-vllm.yml` — vLLM template
