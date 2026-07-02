#!/usr/bin/env python3
"""
compare_corpora.py — Diversity comparison between two abstract-input corpora.

Single-API mode (pass two .jsonl files directly):
    python compare_corpora.py corpus_torch/torch.conv2d/abstract_inputs.jsonl \\
                              corpus_torch_random/torch.conv2d/abstract_inputs.jsonl \\
                              --label1 z3 --label2 random --out results/

Multi-API mode (pass two corpus root directories):
    python compare_corpora.py corpus_torch corpus_torch_random \\
                              --label1 z3 --label2 random \\
                              --apis-file torch_variations.txt \\
                              --out comparison_results/

    When --apis-file is omitted every API subdirectory present in BOTH roots
    is compared automatically.

Metrics
-------
unique_ratio          fraction of distinct samples (exact JSON match)
mean_pairwise_dist    mean normalised Manhattan distance over random pairs
mean_feature_entropy  mean Shannon entropy (bits) across all extracted features
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_CLIP = 1e12   # clip int64 extremes before converting to float


def _extract_numbers(v):
    """Recursively collect all numeric leaf values from a JSON value."""
    if isinstance(v, (int, float)):
        return [float(np.clip(v, -_CLIP, _CLIP))]
    if isinstance(v, list):
        result = []
        for item in v:
            result.extend(_extract_numbers(item))
        return result
    return []


def sample_to_vector(sample: dict) -> list:
    """Flatten a JSON abstract-input dict to a list of floats (key-sorted)."""
    values = []
    for key in sorted(sample.keys()):
        values.extend(_extract_numbers(sample[key]))
    return values


def load_jsonl(path: str) -> list:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        print(f"Warning: {path} is empty.", file=sys.stderr)
    return samples


def _pad(vecs: list, width: int) -> np.ndarray:
    mat = np.zeros((len(vecs), width), dtype=np.float64)
    for i, v in enumerate(vecs):
        mat[i, :len(v)] = v
    return mat


def samples_to_aligned_matrices(samples1: list, samples2: list):
    """Build feature matrices for both corpora with the same column width."""
    vecs1 = [sample_to_vector(s) for s in samples1]
    vecs2 = [sample_to_vector(s) for s in samples2]
    width = max(
        max(len(v) for v in vecs1) if vecs1 else 0,
        max(len(v) for v in vecs2) if vecs2 else 0,
    )
    return _pad(vecs1, width), _pad(vecs2, width)


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def unique_ratio(samples: list) -> float:
    """Fraction of samples that are distinct (exact JSON key-value match)."""
    keys = [json.dumps(s, sort_keys=True) for s in samples]
    return len(set(keys)) / len(keys)


def mean_pairwise_distance(mat: np.ndarray, n_pairs: int = 5000,
                           rng=None) -> float:
    """Normalised mean Manhattan distance over randomly drawn pairs.

    Each dimension is normalised by its observed range so that wide integer
    axes (e.g. value ranges) don't dominate narrow categorical ones (dtypes).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(mat)
    if n < 2:
        return 0.0
    k = min(n_pairs, n * (n - 1) // 2)
    idx = rng.integers(0, n, (k * 3, 2))
    idx = idx[idx[:, 0] != idx[:, 1]][:k]
    diffs = np.abs(mat[idx[:, 0]] - mat[idx[:, 1]])
    col_range = mat.max(axis=0) - mat.min(axis=0)
    col_range[col_range == 0] = 1.0
    return float((diffs / col_range).sum(axis=1).mean())


def feature_entropy(mat: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) for each feature column."""
    entropies = []
    for col in mat.T:
        _, counts = np.unique(col, return_counts=True)
        if len(counts) <= 1:
            entropies.append(0.0)
            continue
        probs = counts / counts.sum()
        entropies.append(float(-np.sum(probs * np.log2(probs + 1e-12))))
    return np.array(entropies)


def mean_entropy(mat: np.ndarray) -> float:
    return float(feature_entropy(mat).mean())


def compute_metrics(samples: list, mat: np.ndarray, n_pairs: int) -> dict:
    return {
        "n":            len(samples),
        "unique_ratio": unique_ratio(samples),
        "mpd":          mean_pairwise_distance(mat, n_pairs=n_pairs),
        "entropy":      mean_entropy(mat),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _safe_project(mat: np.ndarray, mean: np.ndarray,
                  components: np.ndarray) -> np.ndarray:
    """Project mat onto components and guarantee a (n, 2) float64 output.

    Pads with a zero column when fewer than 2 principal components are
    available (degenerate case: single sample, or all features constant).
    Replaces any NaN / Inf produced by the projection with 0.
    """
    proj = np.nan_to_num((mat - mean) @ components.T, nan=0.0, posinf=0.0, neginf=0.0)
    if proj.ndim == 1:
        proj = proj[:, np.newaxis]
    if proj.shape[1] < 2:
        pad = np.zeros((proj.shape[0], 2 - proj.shape[1]), dtype=proj.dtype)
        proj = np.hstack([proj, pad])
    return proj


def _pca_fit_combined(mat1: np.ndarray, mat2: np.ndarray):
    """Return (mean, components) where components has shape (min(2,d), d).

    _safe_project always pads the projection to 2 columns, so callers do
    not need to worry about fewer than 2 principal components.
    """
    combined = np.vstack([mat1, mat2]).astype(np.float64)
    mean = combined.mean(axis=0)
    centered = combined - mean
    d = combined.shape[1]
    n_comp = min(2, d)
    if d == 0 or np.allclose(centered, 0):
        # No variance: use identity-style fallback with correct shape (n_comp, d)
        return mean, np.eye(n_comp, d)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:n_comp]           # shape (n_comp, d)
        if components.shape[0] < n_comp:   # SVD returned fewer rows than expected
            pad = np.zeros((n_comp - components.shape[0], d))
            components = np.vstack([components, pad])
    except np.linalg.LinAlgError:
        components = np.eye(n_comp, d)
    return mean, components


def _safe_ax_limits(ax, proj: np.ndarray, axis: int, pad: float = 0.05) -> None:
    """Set explicit axis limits so matplotlib never sees a zero-range axis."""
    vals = proj[:, axis]
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    margin = (hi - lo) * pad
    setter = ax.set_xlim if axis == 0 else ax.set_ylim
    setter(lo - margin, hi + margin)


def plot_scatter(mat1: np.ndarray, mat2: np.ndarray,
                 label1: str, label2: str,
                 metrics1: dict, metrics2: dict,
                 title: str, out_path: str) -> None:
    """Side-by-side PCA scatter; PCA fitted on the union of both corpora."""
    mean, components = _pca_fit_combined(mat1, mat2)
    p1 = _safe_project(mat1, mean, components)
    p2 = _safe_project(mat2, mean, components)

    colors = ["#4C72B0", "#DD8452"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))   # no sharex/sharey — set manually
    all_proj = np.vstack([p1, p2])
    for axis_i in range(2):
        vals = all_proj[:, axis_i]
        lo, hi = float(vals.min()), float(vals.max())
        if lo == hi:
            lo -= 1.0; hi += 1.0
        margin = (hi - lo) * 0.05
        for ax in axes:
            (ax.set_xlim if axis_i == 0 else ax.set_ylim)(lo - margin, hi + margin)

    for ax, proj, label, color, m in zip(
            axes, [p1, p2], [label1, label2], colors, [metrics1, metrics2]):
        ax.scatter(proj[:, 0], proj[:, 1], alpha=0.35, s=8, color=color,
                   rasterized=True)
        subtitle = (f"n={m['n']}  uniq={m['unique_ratio']:.3f}  "
                    f"mpd={m['mpd']:.3f}  H={m['entropy']:.2f}b")
        ax.set_title(f"{label}\n{subtitle}", fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_entropy_bars(mat1: np.ndarray, mat2: np.ndarray,
                      label1: str, label2: str,
                      title: str, out_path: str) -> None:
    """Bar chart of per-feature entropy for both corpora."""
    e1 = feature_entropy(mat1)
    e2 = feature_entropy(mat2)
    n_features = max(len(e1), len(e2))

    if n_features == 0:
        # Nothing to plot — write a blank figure with a message
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No features extracted", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    e1 = np.pad(e1, (0, n_features - len(e1)))
    e2 = np.pad(e2, (0, n_features - len(e2)))
    # Replace any NaN that slipped through
    e1 = np.nan_to_num(e1)
    e2 = np.nan_to_num(e2)

    x = np.arange(n_features)
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, n_features * 0.25), 4))
    ax.bar(x - width / 2, e1, width, label=label1, color="#4C72B0", alpha=0.8)
    ax.bar(x + width / 2, e2, width, label=label2, color="#DD8452", alpha=0.8)
    # Explicit y-limits so matplotlib never computes a NaN range
    ymax = max(float(e1.max()), float(e2.max()), 0.1)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlabel("Feature index (flattened abstract input)")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_summary(rows: list, label1: str, label2: str,
                       metric: str, ylabel: str,
                       out_path: str) -> None:
    """Grouped bar chart comparing one metric across all APIs."""
    apis = [r["api"] for r in rows]
    v1    = np.nan_to_num([r[f"{metric}_1"] for r in rows])
    v2    = np.nan_to_num([r[f"{metric}_2"] for r in rows])
    delta = v2 - v1

    x = np.arange(len(apis))
    width = 0.3
    fig, ax = plt.subplots(figsize=(max(10, len(apis) * 0.6), 5))
    ax.bar(x - width, v1,    width, label=label1, color="#4C72B0", alpha=0.85)
    ax.bar(x,         v2,    width, label=label2, color="#DD8452", alpha=0.85)
    ax.bar(x + width, delta, width, label="delta (2−1)",
           color=["#2ca02c" if d >= 0 else "#d62728" for d in delta],
           alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(apis, rotation=45, ha="right", fontsize=7)
    # Explicit y-limits
    all_vals = np.concatenate([v1, v2, delta])
    ylo = min(float(all_vals.min()), 0)
    yhi = max(float(all_vals.max()), 0.01)
    margin = (yhi - ylo) * 0.1
    ax.set_ylim(ylo - margin, yhi + margin)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: {label1} vs {label2}")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Single-API comparison
# ---------------------------------------------------------------------------

def compare_one(path1: str, path2: str,
                label1: str, label2: str,
                api_name: str, out_dir: str,
                n_pairs: int) -> dict | None:
    """Compare one API's two corpora. Returns a metrics dict or None on error."""
    if not os.path.exists(path1):
        print(f"  [skip] {api_name}: missing {path1}", file=sys.stderr)
        return None
    if not os.path.exists(path2):
        print(f"  [skip] {api_name}: missing {path2}", file=sys.stderr)
        return None

    samples1 = load_jsonl(path1)
    samples2 = load_jsonl(path2)
    if not samples1 or not samples2:
        print(f"  [skip] {api_name}: one or both corpora empty.", file=sys.stderr)
        return None

    mat1, mat2 = samples_to_aligned_matrices(samples1, samples2)
    m1 = compute_metrics(samples1, mat1, n_pairs)
    m2 = compute_metrics(samples2, mat2, n_pairs)

    os.makedirs(out_dir, exist_ok=True)
    stem = f"{label1}_vs_{label2}"

    plot_scatter(mat1, mat2, label1, label2, m1, m2,
                 f"{api_name}: {label1} vs {label2}",
                 os.path.join(out_dir, f"{stem}_scatter.png"))
    plot_entropy_bars(mat1, mat2, label1, label2,
                      f"{api_name}: per-feature entropy",
                      os.path.join(out_dir, f"{stem}_entropy.png"))

    report_path = os.path.join(out_dir, f"{stem}_report.txt")
    _write_single_report(report_path, api_name, label1, m1, label2, m2)

    return {
        "api":          api_name,
        "n_1":          m1["n"],
        "n_2":          m2["n"],
        "unique_ratio_1": m1["unique_ratio"],
        "unique_ratio_2": m2["unique_ratio"],
        "mpd_1":        m1["mpd"],
        "mpd_2":        m2["mpd"],
        "entropy_1":    m1["entropy"],
        "entropy_2":    m2["entropy"],
    }


def _write_single_report(path: str, api: str,
                         label1: str, m1: dict,
                         label2: str, m2: dict) -> None:
    lines = [
        "=" * 60,
        f"API: {api}",
        f"Diversity comparison: {label1}  vs  {label2}",
        "=" * 60,
    ]
    for label, m in [(label1, m1), (label2, m2)]:
        lines += [
            f"\n--- {label} (n={m['n']}) ---",
            f"  unique ratio              : {m['unique_ratio']:.4f}",
            f"  mean pairwise distance    : {m['mpd']:.4f}",
            f"  mean feature entropy      : {m['entropy']:.4f} bits",
        ]
    lines += [
        f"\n--- delta ({label2} − {label1}) ---",
        f"  unique ratio              : {m2['unique_ratio'] - m1['unique_ratio']:+.4f}",
        f"  mean pairwise distance    : {m2['mpd'] - m1['mpd']:+.4f}",
        f"  mean feature entropy      : {m2['entropy'] - m1['entropy']:+.4f} bits",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_COL = {
    "api":            ("API",               36),
    "n_1":            ("n(1)",               7),
    "n_2":            ("n(2)",               7),
    "unique_ratio_1": ("uniq(1)",            8),
    "unique_ratio_2": ("uniq(2)",            8),
    "mpd_1":          ("mpd(1)",             8),
    "mpd_2":          ("mpd(2)",             8),
    "entropy_1":      ("H(1) b",             8),
    "entropy_2":      ("H(2) b",             8),
}


def _fmt(key: str, val) -> str:
    w = _COL[key][1]
    if isinstance(val, float):
        return f"{val:{w}.4f}"
    return f"{val:{w}}"


def print_table(rows: list, label1: str, label2: str) -> None:
    header_parts = []
    sep_parts = []
    for key, (title, w) in _COL.items():
        h = title.replace("(1)", f"({label1})").replace("(2)", f"({label2})")
        header_parts.append(f"{h:>{w}}")
        sep_parts.append("-" * w)
    print("\n" + "  ".join(header_parts))
    print("  ".join(sep_parts))
    for row in rows:
        print("  ".join(_fmt(k, row[k]) for k in _COL))
    # Averages footer
    num_rows = len(rows)
    if num_rows > 1:
        print("  ".join(sep_parts))
        avg = {"api": "AVERAGE"}
        for key in list(_COL)[1:]:
            avg[key] = sum(r[key] for r in rows) / num_rows
        print("  ".join(_fmt(k, avg[k]) for k in _COL))
    print()


# ---------------------------------------------------------------------------
# Multi-API summary outputs
# ---------------------------------------------------------------------------

def write_summary_csv(rows: list, label1: str, label2: str, path: str) -> None:
    fieldnames = list(_COL.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_report(rows: list, label1: str, label2: str, path: str) -> None:
    lines = [
        "=" * 70,
        f"Multi-API diversity comparison: {label1}  vs  {label2}",
        f"APIs compared: {len(rows)}",
        "=" * 70,
    ]
    for row in rows:
        api = row["api"]
        d_uniq = row["unique_ratio_2"] - row["unique_ratio_1"]
        d_mpd  = row["mpd_2"] - row["mpd_1"]
        d_ent  = row["entropy_2"] - row["entropy_1"]
        lines.append(
            f"\n{api}\n"
            f"  n: {row['n_1']} / {row['n_2']}"
            f"  unique_ratio: {row['unique_ratio_1']:.4f} / {row['unique_ratio_2']:.4f}  ({d_uniq:+.4f})\n"
            f"  mpd:          {row['mpd_1']:.4f} / {row['mpd_2']:.4f}  ({d_mpd:+.4f})\n"
            f"  entropy:      {row['entropy_1']:.4f} / {row['entropy_2']:.4f} bits  ({d_ent:+.4f})"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _jsonl_path(root: str, api: str) -> str:
    return os.path.join(root, api, "abstract_inputs.jsonl")


def main():
    parser = argparse.ArgumentParser(
        description="Compare diversity of two abstract-input corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("corpus1",
                        help="Path to first  abstract_inputs.jsonl  OR  corpus root directory")
    parser.add_argument("corpus2",
                        help="Path to second abstract_inputs.jsonl  OR  corpus root directory")
    parser.add_argument("--label1", default="z3",
                        help="Display label for corpus1  (default: z3)")
    parser.add_argument("--label2", default="random",
                        help="Display label for corpus2  (default: random)")
    parser.add_argument("--out", default="comparison_results",
                        help="Output directory  (default: comparison_results)")
    parser.add_argument("--n-pairs", type=int, default=5000,
                        help="Pairs for mean pairwise distance  (default: 5000)")
    parser.add_argument("--apis-file", default=None,
                        help="(Multi-API mode) text file listing API variant names, "
                             "one per line.  When omitted, all subdirectories present "
                             "in both corpus roots are compared.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Determine mode: single-API (two .jsonl files) vs multi-API (dirs)  #
    # ------------------------------------------------------------------ #
    is_single = (args.corpus1.endswith(".jsonl") or
                 args.corpus2.endswith(".jsonl") or
                 os.path.isfile(args.corpus1))

    if is_single:
        # ---- Single-API mode ----------------------------------------- #
        samples1 = load_jsonl(args.corpus1)
        samples2 = load_jsonl(args.corpus2)
        if not samples1 or not samples2:
            print("Error: one or both corpora are empty.", file=sys.stderr)
            sys.exit(1)

        mat1, mat2 = samples_to_aligned_matrices(samples1, samples2)
        m1 = compute_metrics(samples1, mat1, args.n_pairs)
        m2 = compute_metrics(samples2, mat2, args.n_pairs)

        # Console summary
        row = {
            "api": os.path.basename(os.path.dirname(args.corpus1)),
            "n_1": m1["n"],           "n_2": m2["n"],
            "unique_ratio_1": m1["unique_ratio"], "unique_ratio_2": m2["unique_ratio"],
            "mpd_1": m1["mpd"],       "mpd_2": m2["mpd"],
            "entropy_1": m1["entropy"], "entropy_2": m2["entropy"],
        }
        print_table([row], args.label1, args.label2)

        stem = f"{args.label1}_vs_{args.label2}"
        scatter_path = os.path.join(args.out, f"{stem}_scatter.png")
        entropy_path = os.path.join(args.out, f"{stem}_entropy.png")
        report_path  = os.path.join(args.out, f"{stem}_report.txt")

        plot_scatter(mat1, mat2, args.label1, args.label2, m1, m2,
                     f"{row['api']}: {args.label1} vs {args.label2}",
                     scatter_path)
        plot_entropy_bars(mat1, mat2, args.label1, args.label2,
                          f"{row['api']}: per-feature entropy",
                          entropy_path)
        _write_single_report(report_path, row["api"],
                             args.label1, m1, args.label2, m2)

        print(f"scatter  → {scatter_path}")
        print(f"entropy  → {entropy_path}")
        print(f"report   → {report_path}")

    else:
        # ---- Multi-API mode ------------------------------------------ #
        if args.apis_file:
            with open(args.apis_file) as f:
                apis = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover: intersection of subdirs present in both roots
            subdirs1 = {d for d in os.listdir(args.corpus1)
                        if os.path.isdir(os.path.join(args.corpus1, d))}
            subdirs2 = {d for d in os.listdir(args.corpus2)
                        if os.path.isdir(os.path.join(args.corpus2, d))}
            apis = sorted(subdirs1 & subdirs2)
            if not apis:
                print("No common API subdirectories found in both corpus roots.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"Auto-discovered {len(apis)} common APIs.")

        rows = []
        for api in apis:
            print(f"  comparing {api} …")
            row = compare_one(
                path1   = _jsonl_path(args.corpus1, api),
                path2   = _jsonl_path(args.corpus2, api),
                label1  = args.label1,
                label2  = args.label2,
                api_name= api,
                out_dir = os.path.join(args.out, api),
                n_pairs = args.n_pairs,
            )
            if row is not None:
                rows.append(row)

        if not rows:
            print("No valid API pairs found.", file=sys.stderr)
            sys.exit(1)

        print_table(rows, args.label1, args.label2)

        stem = f"{args.label1}_vs_{args.label2}"

        # Per-metric summary bar charts
        for metric, ylabel in [
            ("unique_ratio", "Unique ratio"),
            ("mpd",          "Mean pairwise distance (normalised)"),
            ("entropy",      "Mean feature entropy (bits)"),
        ]:
            plot_multi_summary(rows, args.label1, args.label2,
                               metric, ylabel,
                               os.path.join(args.out, f"{stem}_{metric}.png"))

        csv_path    = os.path.join(args.out, f"{stem}_summary.csv")
        report_path = os.path.join(args.out, f"{stem}_summary.txt")
        write_summary_csv(rows, args.label1, args.label2, csv_path)
        write_summary_report(rows, args.label1, args.label2, report_path)

        print(f"\nPer-API plots written to {args.out}/<api>/")
        print(f"Summary CSV     → {csv_path}")
        print(f"Summary report  → {report_path}")
        print(f"Summary charts  → {args.out}/{stem}_{{unique_ratio,mpd,entropy}}.png")


if __name__ == "__main__":
    main()
