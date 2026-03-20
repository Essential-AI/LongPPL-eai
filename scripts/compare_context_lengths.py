"""
Compare LongPPL metrics across context lengths (32K vs 64K vs 128K).

Reads per-source JSON result files and produces a comparison table showing
how key metrics change as context length increases.

Usage:
    python scripts/compare_context_lengths.py \
        --results-dir /tmp \
        --output /tmp/context_length_comparison.md
"""

import argparse
import json
import os
import sys

import numpy as np


SOURCES = [
    "science-and-math",
    "library",
    "ia-ascm",
    "programming",
    "arxiv",
    "eai-crawl-journals",
]

SHORT_NAMES = {
    "science-and-math": "Sci-Math",
    "library": "Library",
    "ia-ascm": "IA-ASCM",
    "programming": "Programming",
    "arxiv": "Arxiv",
    "eai-crawl-journals": "EAI-Journals",
}

KEY_METRICS = [
    "key_token_ratio",
    "mean_context_benefit",
    "median_context_benefit",
    "frac_positive_benefit",
    "context_benefit_p90",
    "context_benefit_near",
    "context_benefit_mid",
    "context_benefit_far",
]


def load_results(path):
    """Load results JSON and return list of per-doc result dicts."""
    with open(path) as f:
        data = json.load(f)
    return data.get("results", [])


def summarize(results, metric):
    """Compute mean of a metric across docs."""
    vals = [r[metric] for r in results if r.get(metric) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-32k-dir", default="/tmp", help="Dir with longppl_32k_*.json files")
    parser.add_argument("--results-128k-dir", default="/tmp", help="Dir with longppl_128k_*.json files")
    parser.add_argument("--output", default="/tmp/context_length_comparison.md")
    args = parser.parse_args()

    # Try to load results for each source at each context length
    all_data = {}  # (source, context_len) -> results

    for src in SOURCES:
        # 32K (100 docs)
        path_32k = os.path.join(args.results_32k_dir, f"longppl_32k_{src}.json")
        if os.path.exists(path_32k):
            all_data[(src, "32K")] = load_results(path_32k)

        # 64K (20 docs)
        path_64k = os.path.join(args.results_128k_dir, f"longppl_64k_{src}.json")
        if os.path.exists(path_64k):
            all_data[(src, "64K")] = load_results(path_64k)

        # 128K (20 docs)
        path_128k = os.path.join(args.results_128k_dir, f"longppl_128k_{src}.json")
        if os.path.exists(path_128k):
            all_data[(src, "128K")] = load_results(path_128k)

    if not all_data:
        print("No results found!")
        sys.exit(1)

    # Build comparison report
    lines = []
    lines.append("# Context Length Comparison: 32K vs 64K vs 128K\n")
    lines.append(f"Sources with data: {len(set(s for s, _ in all_data))}\n")

    # Per-metric comparison tables
    for metric in KEY_METRICS:
        lines.append(f"\n## {metric}\n")
        lines.append(f"| Source | 32K (n=100) | 64K (n=20) | 128K (n=20) | 32K→128K Change |")
        lines.append(f"|--------|-------------|------------|-------------|-----------------|")

        for src in SOURCES:
            short = SHORT_NAMES[src]
            vals = {}
            for cl in ["32K", "64K", "128K"]:
                if (src, cl) in all_data:
                    vals[cl] = summarize(all_data[(src, cl)], metric)

            v32 = f"{vals['32K']:.4f}" if vals.get("32K") is not None else "—"
            v64 = f"{vals['64K']:.4f}" if vals.get("64K") is not None else "—"
            v128 = f"{vals['128K']:.4f}" if vals.get("128K") is not None else "—"

            if vals.get("32K") is not None and vals.get("128K") is not None and vals["32K"] != 0:
                change = (vals["128K"] - vals["32K"]) / abs(vals["32K"]) * 100
                change_str = f"{change:+.1f}%"
            elif vals.get("32K") is not None and vals.get("64K") is not None and vals["32K"] != 0:
                change = (vals["64K"] - vals["32K"]) / abs(vals["32K"]) * 100
                change_str = f"{change:+.1f}% (64K)"
            else:
                change_str = "—"

            lines.append(f"| {short} | {v32} | {v64} | {v128} | {change_str} |")

    # Position analysis comparison
    lines.append("\n## Position Analysis: Near/Mid/Far Ratio\n")
    lines.append("Shows far/near ratio — higher means context benefit grows more with distance.\n")
    lines.append(f"| Source | 32K far/near | 128K far/near | Change |")
    lines.append(f"|--------|-------------|---------------|--------|")

    for src in SOURCES:
        short = SHORT_NAMES[src]
        ratios = {}
        for cl in ["32K", "128K", "64K"]:
            if (src, cl) in all_data:
                near = summarize(all_data[(src, cl)], "context_benefit_near")
                far = summarize(all_data[(src, cl)], "context_benefit_far")
                if near and near != 0:
                    ratios[cl] = far / near

        v32 = f"{ratios['32K']:.2f}x" if ratios.get("32K") is not None else "—"
        v128 = f"{ratios['128K']:.2f}x" if ratios.get("128K") is not None else (
            f"{ratios['64K']:.2f}x (64K)" if ratios.get("64K") is not None else "—"
        )

        if ratios.get("32K") and ratios.get("128K"):
            change = ratios["128K"] - ratios["32K"]
            change_str = f"{change:+.2f}x"
        elif ratios.get("32K") and ratios.get("64K"):
            change = ratios["64K"] - ratios["32K"]
            change_str = f"{change:+.2f}x (64K)"
        else:
            change_str = "—"

        lines.append(f"| {short} | {v32} | {v128} | {change_str} |")

    # Summary statistics
    lines.append("\n## Key Findings\n")

    # Compute aggregate changes
    ktr_changes = []
    mcb_changes = []
    for src in SOURCES:
        for long_cl in ["128K", "64K"]:
            if (src, "32K") in all_data and (src, long_cl) in all_data:
                ktr_32 = summarize(all_data[(src, "32K")], "key_token_ratio")
                ktr_long = summarize(all_data[(src, long_cl)], "key_token_ratio")
                mcb_32 = summarize(all_data[(src, "32K")], "mean_context_benefit")
                mcb_long = summarize(all_data[(src, long_cl)], "mean_context_benefit")
                if ktr_32 and ktr_long:
                    ktr_changes.append((src, long_cl, ktr_32, ktr_long))
                if mcb_32 and mcb_long:
                    mcb_changes.append((src, long_cl, mcb_32, mcb_long))
                break  # Use the longest available

    if ktr_changes:
        lines.append("### Key Token Ratio changes:\n")
        for src, cl, v32, vlong in sorted(ktr_changes, key=lambda x: x[3]/x[2] if x[2] else 0, reverse=True):
            pct = (vlong - v32) / v32 * 100 if v32 else 0
            lines.append(f"- **{SHORT_NAMES[src]}**: {v32:.4f} → {vlong:.4f} ({pct:+.0f}%, {cl})")

    if mcb_changes:
        lines.append("\n### MCB changes:\n")
        for src, cl, v32, vlong in sorted(mcb_changes, key=lambda x: x[3]/x[2] if x[2] else 0, reverse=True):
            pct = (vlong - v32) / v32 * 100 if v32 else 0
            lines.append(f"- **{SHORT_NAMES[src]}**: {v32:.4f} → {vlong:.4f} ({pct:+.0f}%, {cl})")

    # Per-doc detail for each source (at longest context)
    lines.append("\n## Per-Doc Distributions at Long Context\n")
    for src in SOURCES:
        for cl in ["128K", "64K"]:
            if (src, cl) in all_data:
                results = all_data[(src, cl)]
                short = SHORT_NAMES[src]
                n = len(results)
                ktrs = [r["key_token_ratio"] for r in results]
                mcbs = [r.get("mean_context_benefit", 0) for r in results if r.get("mean_context_benefit") is not None]

                lines.append(f"### {short} ({cl}, n={n})\n")
                if ktrs:
                    lines.append(f"- KTR: mean={np.mean(ktrs):.4f}, std={np.std(ktrs):.4f}, "
                                f"min={np.min(ktrs):.4f}, max={np.max(ktrs):.4f}")
                if mcbs:
                    lines.append(f"- MCB: mean={np.mean(mcbs):.4f}, std={np.std(mcbs):.4f}, "
                                f"min={np.min(mcbs):.4f}, max={np.max(mcbs):.4f}")
                lines.append("")
                break

    report = "\n".join(lines)
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
