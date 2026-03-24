"""
Profile where time is spent in the forward pass: attention vs linear layers.
Test batching at each context length to find the sweet spot.

Loads LLaMA-3.1-8B (hardcoded) on a single GPU and runs instrumented forward
passes at context lengths from 4K to 128K. Uses PyTorch hooks to measure time
in attention vs MLP sublayers, then tests batching (batch_size=1..4) at each
context length to find the throughput sweet spot.

No CLI args — all parameters are hardcoded. Requires a single MI300X GPU.

Output: prints a table of context_length x batch_size with tokens/sec, time/doc,
and attention vs MLP time breakdown.

Usage:
    python scripts/profile_bench.py
"""

import subprocess
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
    return model


def profile_layers(model, device, seq_len, batch_size=1):
    """
    Profile a single forward pass, measuring time in attention vs MLP.
    Uses PyTorch hooks to instrument each layer.
    """
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), dtype=torch.long, device=device)

    layer_times = {}
    hooks = []

    def make_hooks(name, module):
        start_time = {}

        def pre_hook(mod, inp):
            torch.cuda.synchronize()
            start_time['t'] = time.time()

        def post_hook(mod, inp, out):
            torch.cuda.synchronize()
            elapsed = time.time() - start_time['t']
            if name not in layer_times:
                layer_times[name] = []
            layer_times[name].append(elapsed)

        hooks.append(module.register_forward_pre_hook(pre_hook))
        hooks.append(module.register_forward_hook(post_hook))

    # Instrument key components of layer 0 (representative)
    layer0 = model.model.layers[0]
    make_hooks("self_attn", layer0.self_attn)
    make_hooks("mlp", layer0.mlp)

    # Also instrument the full model.model (all transformer layers) and lm_head
    make_hooks("all_layers", model.model)
    make_hooks("lm_head", model.lm_head)

    # Warmup
    with torch.no_grad():
        model(input_ids)
    layer_times.clear()

    # Timed run (3 iterations)
    n_iters = 3
    for _ in range(n_iters):
        with torch.no_grad():
            model(input_ids)

    # Clean up hooks
    for h in hooks:
        h.remove()
    del input_ids

    # Average across iterations
    result = {}
    for name, times in layer_times.items():
        result[name] = np.mean(times)

    return result


def bench_batch(model, device, seq_len, batch_size, n_iters=3):
    """Run forward pass and return median per-doc time."""
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
        result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                                capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().split("\n"):
            if "GPU[0]" in line:
                print(line.strip())
    except Exception:
        pass
    print("=" * 60 + "\n")

    model = load_model(model_name)
    device = next(model.parameters()).device

    # Part 1: Profile attention vs MLP time at different context lengths
    print("=" * 70)
    print("PART 1: TIME BREAKDOWN (attention vs MLP) — batch=1")
    print("=" * 70)
    print(f"{'seq_len':>8}  {'total':>8}  {'attn_pct':>9}  {'mlp_pct':>8}  {'lm_head_pct':>11}  {'attn_1layer':>11}  {'mlp_1layer':>10}")
    print("-" * 80)

    for seq_len in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        try:
            times = profile_layers(model, device, seq_len, batch_size=1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{seq_len:>8}  OOM")
            break

        total = times.get("all_layers", 0) + times.get("lm_head", 0)
        # 32 layers total, we measured layer 0
        attn_total = times.get("self_attn", 0) * 32
        mlp_total = times.get("mlp", 0) * 32
        lm_head = times.get("lm_head", 0)

        attn_pct = 100 * attn_total / total if total > 0 else 0
        mlp_pct = 100 * mlp_total / total if total > 0 else 0
        lm_head_pct = 100 * lm_head / total if total > 0 else 0

        print(f"{seq_len:>8}  {total:>7.3f}s  {attn_pct:>8.1f}%  {mlp_pct:>7.1f}%  {lm_head_pct:>10.1f}%  "
              f"{times.get('self_attn', 0)*1000:>9.1f}ms  {times.get('mlp', 0)*1000:>8.1f}ms")

        torch.cuda.empty_cache()

    # Part 2: Batching sweep — find the sweet spot
    print(f"\n{'=' * 70}")
    print("PART 2: BATCHING SWEEP")
    print(f"{'=' * 70}")
    print(f"{'seq_len':>8}  {'batch':>5}  {'total_s':>8}  {'per_doc_s':>10}  {'speedup':>8}  {'tok/s':>10}")
    print("-" * 60)

    for seq_len in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        baseline = None
        for bs in [1, 2, 4, 8, 16, 32, 64]:
            try:
                t = bench_batch(model, device, seq_len, bs, n_iters=3)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break

            per_doc = t / bs
            tps = (bs * seq_len) / t
            if baseline is None:
                baseline = per_doc
            speedup = baseline / per_doc

            print(f"{seq_len:>8}  {bs:>5}  {t:>8.3f}  {per_doc:>10.4f}  {speedup:>7.2f}x  {tps:>10.0f}")

        print()


if __name__ == "__main__":
    main()
