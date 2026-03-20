"""
Analyze extended LongPPL metrics across data sources.

Reads per-source JSON results from the extended metrics run, computes
cross-source comparisons, metric correlations, and position analysis.
Outputs a summary report and CSV for further analysis.

Usage:
    # After downloading results:
    python scripts/analyze_extended_metrics.py --results-dir /tmp/longppl_extended_100
    python scripts/analyze_extended_metrics.py --results-dir /tmp/longppl_extended_val
"""

import argparse
import json
import os
import sys

import numpy as np


SOURCES = ["arxiv", "programming", "eai-crawl-journals", "science-and-math", "ia-ascm", "library"]

METRICS = [
    "key_token_ratio",
    "mean_context_benefit",
    "median_context_benefit",
    "context_benefit_std",
    "frac_positive_benefit",
    "weighted_context_benefit",
    "mean_key_improvement",
    "context_benefit_near",
    "context_benefit_mid",
    "context_benefit_far",
]

SHORT_NAMES = {
    "key_token_ratio": "KTR",
    "mean_context_benefit": "MCB",
    "median_context_benefit": "MedCB",
    "context_benefit_std": "CB_Std",
    "frac_positive_benefit": "FracPos",
    "weighted_context_benefit": "WCB",
    "mean_key_improvement": "KeyImp",
    "context_benefit_near": "CB_Near",
    "context_benefit_mid": "CB_Mid",
    "context_benefit_far": "CB_Far",
}


def load_results(results_dir):
    """Load per-source results JSONs into {source: [doc_results]}."""
    data = {}
    for src in SOURCES:
        path = os.path.join(results_dir, f"{src}.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {src}")
            continue
        with open(path) as f:
            raw = json.load(f)
        data[src] = raw["results"]
        print(f"  {src}: {len(raw['results'])} docs")
    return data


def source_metric_means(data):
    """Compute mean of each metric per source. Returns {source: {metric: mean}}."""
    out = {}
    for src, docs in data.items():
        out[src] = {}
        for m in METRICS:
            vals = [d[m] for d in docs if d.get(m) is not None]
            out[src][m] = float(np.mean(vals)) if vals else None
    return out


def print_ranking_table(means):
    """Print sources ranked by each metric."""
    print("\n" + "=" * 90)
    print("SOURCE RANKINGS BY METRIC")
    print("=" * 90)

    for m in METRICS:
        pairs = [(src, means[src][m]) for src in means if means[src][m] is not None]
        pairs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  {SHORT_NAMES[m]} ({m}):")
        for rank, (src, val) in enumerate(pairs, 1):
            print(f"    {rank}. {src:<25s} {val:.4f}")


def compute_correlation_matrix(data):
    """Compute pairwise Pearson correlations across all docs (pooled)."""
    all_docs = []
    for src, docs in data.items():
        for d in docs:
            d_copy = dict(d)
            d_copy["_source"] = src
            all_docs.append(d_copy)

    n = len(METRICS)
    corr = np.full((n, n), np.nan)

    for i, m1 in enumerate(METRICS):
        for j, m2 in enumerate(METRICS):
            v1 = [d[m1] for d in all_docs if d.get(m1) is not None and d.get(m2) is not None]
            v2 = [d[m2] for d in all_docs if d.get(m1) is not None and d.get(m2) is not None]
            if len(v1) >= 3:
                corr[i, j] = float(np.corrcoef(v1, v2)[0, 1])

    return corr


def print_correlation_matrix(corr):
    """Print correlation matrix as a table."""
    print("\n" + "=" * 90)
    print("METRIC CORRELATION MATRIX (Pearson, pooled across all sources)")
    print("=" * 90)

    header = "         " + "  ".join(f"{SHORT_NAMES[m]:>7s}" for m in METRICS)
    print(header)
    for i, m in enumerate(METRICS):
        row = f"{SHORT_NAMES[m]:>8s} "
        for j in range(len(METRICS)):
            if np.isnan(corr[i, j]):
                row += "      - "
            else:
                row += f" {corr[i, j]:>6.2f} "
        print(row)


def position_analysis(data):
    """Analyze whether context benefit changes with distance from trunc_len boundary."""
    print("\n" + "=" * 90)
    print("POSITION ANALYSIS: Context benefit by distance from short-context boundary")
    print("=" * 90)
    print(f"\n  {'Source':<25s} {'Near':>8s} {'Mid':>8s} {'Far':>8s}  {'Trend':>12s}")
    print("  " + "-" * 65)

    for src in SOURCES:
        if src not in data:
            continue
        docs = data[src]
        near = [d["context_benefit_near"] for d in docs if d.get("context_benefit_near") is not None]
        mid = [d["context_benefit_mid"] for d in docs if d.get("context_benefit_mid") is not None]
        far = [d["context_benefit_far"] for d in docs if d.get("context_benefit_far") is not None]
        if not near:
            continue

        mn, mm, mf = np.mean(near), np.mean(mid), np.mean(far)
        if mf > mn * 1.1:
            trend = "increasing"
        elif mf < mn * 0.9:
            trend = "decreasing"
        else:
            trend = "flat"
        print(f"  {src:<25s} {mn:>8.4f} {mm:>8.4f} {mf:>8.4f}  {trend:>12s}")


def diffuse_vs_sparse_analysis(data):
    """Identify sources where MCB tells a different story than key_token_ratio."""
    print("\n" + "=" * 90)
    print("DIFFUSE vs SPARSE: Sources that rank differently on MCB vs Key Token Ratio")
    print("=" * 90)

    means = source_metric_means(data)

    ktr_rank = sorted(means.keys(), key=lambda s: means[s].get("key_token_ratio", 0), reverse=True)
    mcb_rank = sorted(means.keys(), key=lambda s: means[s].get("mean_context_benefit", 0), reverse=True)

    ktr_pos = {s: i for i, s in enumerate(ktr_rank)}
    mcb_pos = {s: i for i, s in enumerate(mcb_rank)}

    print(f"\n  {'Source':<25s} {'KTR Rank':>10s} {'MCB Rank':>10s} {'Shift':>8s}")
    print("  " + "-" * 55)
    for src in SOURCES:
        if src not in means:
            continue
        shift = ktr_pos[src] - mcb_pos[src]
        arrow = f"+{shift}" if shift > 0 else str(shift)
        print(f"  {src:<25s} {ktr_pos[src]+1:>10d} {mcb_pos[src]+1:>10d} {arrow:>8s}")

    print("\n  Positive shift = source ranks higher on MCB than KTR (diffuse benefit)")
    print("  Negative shift = source ranks higher on KTR than MCB (sparse/spiky benefit)")


def filtering_recommendations(data):
    """Suggest filtering thresholds for data curation."""
    print("\n" + "=" * 90)
    print("FILTERING RECOMMENDATIONS FOR DATA CURATION")
    print("=" * 90)

    all_docs = []
    for src, docs in data.items():
        for d in docs:
            d_copy = dict(d)
            d_copy["_source"] = src
            all_docs.append(d_copy)

    # For each candidate metric, show what fraction of docs pass various thresholds
    candidate_metrics = ["key_token_ratio", "mean_context_benefit", "frac_positive_benefit", "weighted_context_benefit"]

    for m in candidate_metrics:
        vals = np.array([d[m] for d in all_docs if d.get(m) is not None])
        if len(vals) == 0:
            continue

        print(f"\n  {m} (mean={np.mean(vals):.4f}, std={np.std(vals):.4f}):")
        percentiles = [25, 50, 75, 90]
        for p in percentiles:
            thresh = np.percentile(vals, p)
            # Show per-source retention at this threshold
            retained = {}
            for src in SOURCES:
                if src not in data:
                    continue
                src_vals = [d[m] for d in data[src] if d.get(m) is not None]
                if src_vals:
                    retained[src] = np.mean(np.array(src_vals) >= thresh)
            src_summary = ", ".join(f"{s}={v:.0%}" for s, v in sorted(retained.items(), key=lambda x: x[1], reverse=True))
            print(f"    p{p} threshold ({thresh:.4f}): {src_summary}")


def write_csv(data, output_path):
    """Write all per-doc results to a flat CSV for external analysis."""
    all_docs = []
    for src, docs in data.items():
        for d in docs:
            row = {"source": src}
            row.update(d)
            all_docs.append(row)

    if not all_docs:
        return

    cols = ["source"] + [k for k in all_docs[0] if k != "source"]
    with open(output_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in all_docs:
            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"\nCSV written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze extended LongPPL metrics")
    parser.add_argument("--results-dir", required=True, help="Directory with per-source JSON files")
    parser.add_argument("--csv-output", default=None, help="Path to write flat CSV (default: <results-dir>/all_docs.csv)")
    args = parser.parse_args()

    if args.csv_output is None:
        args.csv_output = os.path.join(args.results_dir, "all_docs.csv")

    print("Loading results...")
    data = load_results(args.results_dir)
    if not data:
        print("ERROR: No results found.")
        sys.exit(1)

    means = source_metric_means(data)

    # 1. Rankings
    print_ranking_table(means)

    # 2. Correlation matrix
    corr = compute_correlation_matrix(data)
    print_correlation_matrix(corr)

    # 3. Position analysis
    position_analysis(data)

    # 4. Diffuse vs sparse
    diffuse_vs_sparse_analysis(data)

    # 5. Filtering recommendations
    filtering_recommendations(data)

    # 6. CSV export
    write_csv(data, args.csv_output)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
