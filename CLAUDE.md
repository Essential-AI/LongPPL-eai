# CLAUDE.md

## Project Overview

LongPPL is the official implementation of the ICLR 2025 paper "What is Wrong with Perplexity for Long-context Language Modeling?" It provides the LongPPL evaluation metric and LongCE training loss for long-context LLMs. Published on PyPI as `longppl`.

## Repository Structure

- `longppl/` — Core package (pip-installable). `longppl.py` has the main `compute_longppl()` API.
- `perplexity/` — Evaluation scripts and precomputed key tokens for offline mode.
- `finetune/` — LongCE fine-tuning scripts with DeepSpeed configs and EABF position patches.

## Setup

```bash
pip install -e .
# or: pip install longppl
```

Dependencies: Python 3.10+, PyTorch 2.3+, transformers >=4.44.0, accelerate, datasets, einops, sentencepiece, protobuf==3.19.6.

## Common Commands

### Run LongPPL evaluation
```bash
cd perplexity && sh run_ppl.sh
```

### Run fine-tuning
```bash
cd finetune && accelerate config && sh train.sh
```

### Install in development mode
```bash
pip install -e .
```

## Key Concepts

- **Online mode**: Dynamically identifies key tokens using an evaluator model at runtime.
- **Offline mode**: Uses precomputed key tokens from `perplexity/key_text/`.
- **Key parameters**: `alpha` (loss discrepancy threshold, default 2.0), `beta` (loss magnitude threshold, default -2.0), `trunc_len` (short context window, default 4096).

## Data Paths (GCS)

All data lives under `gs://consus-dataproc/ocr/`.

### Tokenized full documents
Input to scoring scripts. Columns: `tokens` (list[int]), `token_len` (int).
To get text, decode tokens with the LLaMA-3.1-8B tokenizer.

| Source | Path |
|--------|------|
| ia-ascm | `gs://consus-dataproc/ocr/ia-ascm/text/tokenized_fulldocs/` (3 batches, ~611K docs) |
| arxiv | `gs://consus-dataproc/ocr/arxiv/text/tokenized_fulldocs/` |
| programming | `gs://consus-dataproc/ocr/programming/text/tokenized_fulldocs_0.424B` (single file) |
| eai-crawl-journals | `gs://consus-dataproc/ocr/eai-crawl-journals/text/tokenized_fulldocs_0.525B` (single file) |
| science-and-math | `gs://consus-dataproc/ocr/science-and-math/text/tokenized_fulldocs_1.190B` |
| library | `gs://consus-dataproc/ocr/library/text/tokenized_fulldocs_8.589B` |

### Raw/filtered text
Pre-tokenization text. Columns include `text`, `xxh3_64_int`, `pdf_gs_path`, etc.

| Source | Path |
|--------|------|
| ia-ascm | `gs://consus-dataproc/ocr/ia-ascm/text/filtered/50chunks/english_entropy_filtered/` |

### Context ladder scores (pairwise metrics)
Output of `scripts/context_ladder.py`. Columns: `source_file`, `row_index`, `doc_len`,
`c_max`, `score_window_start`, `time_s`, plus pairwise metrics `{mcb,ktr,frac_pos,cb_p90}_{c1}v{c2}`.
Join back to tokenized parquets using `(source_file, row_index)` where `source_file` is the
**basename** of the tokenized parquet file.

| Source | Path |
|--------|------|
| ia-ascm 32K ladder | `gs://consus-dataproc/ocr/ia-ascm/scores/context_ladder_32k/` (~98K docs) |
| per-source ladder | `gs://consus-dataproc/ocr/{source}/scores/context_ladder/` |

### LongPPL scores
Output of `scripts/score_partition.py` / `scripts/merge_scores.py`.

| Source | Path |
|--------|------|
| per-source merged | `gs://consus-dataproc/ocr/{source}/scores/longppl/longppl_all_{source}.parquet` |

## Code Conventions

- Pure Python, no type hints, no linter configs.
- PyTorch + Hugging Face Transformers throughout.
- Functional style with module-level helper functions.
- No test suite — validation is done via benchmark evaluation scripts.
- Apache 2.0 license.
