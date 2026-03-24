"""
Micro-benchmark: measure forward pass throughput at different batch sizes
and context lengths on a single GPU.

Loads LLaMA-3.1-8B (hardcoded) and runs forward passes with random input at
context lengths {4K, 8K, 16K, 32K} and batch sizes {1, 2, 4}. Reports
tokens/sec and time/doc to show how batching affects throughput.

No CLI args — all parameters are hardcoded. Requires a single MI300X GPU.

Output: prints a table of context_length x batch_size with tokens/sec and
time per document.

Usage:
    python scripts/batch_microbench.py
"""

import subprocess
import sys
import time

import torch
import numpy as np


def load_model(model_name):
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


def bench(model, device, seq_len, batch_size, n_iters=3):
    """Run n_iters forward passes and return median time."""
    # Use random tokens (not zeros — zeros can trigger degenerate kernel paths)
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        model(input_ids)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model(input_ids)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    del input_ids
    torch.cuda.empty_cache()
    return np.median(times)


def main():
    model_name = "meta-llama/Meta-Llama-3.1-8B"

    # GPU info
    print("=" * 60)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            if "GPU[0]" in line:
                print(line.strip())
    except Exception as e:
        print(f"Could not query GPU: {e}")
    print("=" * 60 + "\n")

    model = load_model(model_name)
    device = next(model.parameters()).device

    # Test matrix: context lengths x batch sizes
    seq_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    batch_sizes = [1, 2, 4, 8, 16, 32]

    print(f"{'seq_len':>8}  {'batch':>5}  {'total_tok':>10}  {'time_s':>8}  {'tok/s':>10}  {'time/doc':>10}  {'speedup':>8}")
    print("-" * 75)

    for seq_len in seq_lengths:
        baseline_per_doc = None
        for bs in batch_sizes:
            total_tokens = bs * seq_len
            # Check if this will OOM (rough estimate: ~2 bytes per token for input,
            # plus KV cache and activations)
            try:
                t = bench(model, device, seq_len, bs, n_iters=3)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{seq_len:>8}  {bs:>5}  {'OOM':>10}")
                break

            tps = total_tokens / t
            per_doc = t / bs
            if baseline_per_doc is None:
                baseline_per_doc = per_doc
            speedup = baseline_per_doc / per_doc

            print(f"{seq_len:>8}  {bs:>5}  {total_tokens:>10}  {t:>8.3f}  {tps:>10.0f}  {per_doc:>10.4f}  {speedup:>7.1f}x")

        print()


if __name__ == "__main__":
    main()
