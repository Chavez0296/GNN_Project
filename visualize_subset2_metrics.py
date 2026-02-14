import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_overall(valid_mean: dict, test_mean: dict, out_path: str):
    keys = ["mrr", "hits@1", "hits@3", "hits@10"]
    x = np.arange(len(keys))
    width = 0.36

    valid_vals = [float(valid_mean[k]) for k in keys]
    test_vals = [float(test_mean[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, valid_vals, width, label="valid_mean")
    ax.bar(x + width / 2, test_vals, width, label="test_mean")

    ax.set_title("Subset 2 DistMult: Overall Metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in keys])
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for i, v in enumerate(valid_vals):
        ax.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(test_vals):
        ax.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_seed_mrr(test_by_seed: list, out_path: str):
    mrr = np.array([float(x["mrr"]) for x in test_by_seed], dtype=float)
    idx = np.arange(1, len(mrr) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(idx, mrr, marker="o", linewidth=1.5)
    ax.axhline(float(mrr.mean()), linestyle="--", color="black", linewidth=1.0, label=f"mean={mrr.mean():.4f}")
    ax.set_title("Subset 2 DistMult: Test MRR by Seed")
    ax.set_xlabel("Seed index")
    ax.set_ylabel("MRR")
    ax.set_xticks(idx)
    ax.set_ylim(max(0.0, float(mrr.min()) - 0.01), min(1.0, float(mrr.max()) + 0.01))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_relation_bar(relation_scores: dict, out_path: str):
    items = sorted((int(k), float(v)) for k, v in relation_scores.items())
    rel_ids = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(rel_ids, vals, color="#54a24b")
    ax.set_title("Subset 2 DistMult: Test MRR by Relation ID")
    ax.set_xlabel("Relation ID")
    ax.set_ylabel("MRR")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_relation_sorted(relation_scores: dict, out_path: str):
    items = sorted(((int(k), float(v)) for k, v in relation_scores.items()), key=lambda x: x[1])
    vals = np.array([v for _, v in items], dtype=float)
    labels = [k for k, _ in items]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(np.arange(len(vals)), vals, color="#e45756")
    ax.set_title("Subset 2 DistMult: Relation MRR (Worst -> Best)")
    ax.set_xlabel("Relation rank index")
    ax.set_ylabel("MRR")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

    if len(vals) >= 3:
        for i in [0, 1, 2, len(vals) - 3, len(vals) - 2, len(vals) - 1]:
            ax.text(i, vals[i] + 0.012, f"r{labels[i]}={vals[i]:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_relation_hist(relation_scores: dict, out_path: str):
    vals = np.array([float(v) for v in relation_scores.values()], dtype=float)
    mean_v = float(vals.mean())
    median_v = float(np.median(vals))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(vals, bins=12, color="#4c78a8", edgecolor="black", alpha=0.85)
    ax.axvline(mean_v, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean_v:.3f}")
    ax.axvline(median_v, color="gray", linestyle=":", linewidth=1.2, label=f"median={median_v:.3f}")
    ax.set_title("Subset 2 DistMult: Relation-Level MRR Distribution")
    ax.set_xlabel("MRR")
    ax.set_ylabel("Number of relations")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary(metrics: dict, out_path: str):
    rel_vals = np.array([float(v) for v in metrics["test_mrr_by_relation_mean"].values()], dtype=float)
    test_by_seed = metrics["test_by_seed"]
    seed_mrr = np.array([float(x["mrr"]) for x in test_by_seed], dtype=float)

    lines = [
        "Subset 2 DistMult Summary",
        "",
        f"Model: {metrics.get('model', 'unknown')}",
        f"Epochs: {metrics.get('config', {}).get('epochs', 'unknown')}",
        "",
        "Test mean metrics:",
        f"- MRR: {metrics['test_mean']['mrr']:.6f}",
        f"- Hits@1: {metrics['test_mean']['hits@1']:.6f}",
        f"- Hits@3: {metrics['test_mean']['hits@3']:.6f}",
        f"- Hits@10: {metrics['test_mean']['hits@10']:.6f}",
        "",
        "Seed stability (test MRR):",
        f"- mean: {seed_mrr.mean():.6f}",
        f"- std: {seed_mrr.std():.6f}",
        f"- min: {seed_mrr.min():.6f}",
        f"- max: {seed_mrr.max():.6f}",
        "",
        "Relation-level (test MRR mean across seeds):",
        f"- count: {len(rel_vals)}",
        f"- mean: {rel_vals.mean():.6f}",
        f"- std: {rel_vals.std():.6f}",
        f"- min: {rel_vals.min():.6f}",
        f"- median: {np.median(rel_vals):.6f}",
        f"- max: {rel_vals.max():.6f}",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Visualize Stage1 Subset2 DistMult metrics")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/stage1_subset2_distmult_metrics.json",
        help="Path to stage1 subset2 distmult metrics JSON",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/figures_subset2_distmult",
        help="Directory to save charts",
    )
    args = parser.parse_args()

    metrics = load_json(args.input)
    os.makedirs(args.outdir, exist_ok=True)

    plot_overall(metrics["valid_mean"], metrics["test_mean"], os.path.join(args.outdir, "subset2_distmult_overall.png"))
    plot_seed_mrr(metrics["test_by_seed"], os.path.join(args.outdir, "subset2_distmult_test_mrr_by_seed.png"))
    plot_relation_bar(metrics["test_mrr_by_relation_mean"], os.path.join(args.outdir, "subset2_distmult_mrr_by_relation_id.png"))
    plot_relation_sorted(metrics["test_mrr_by_relation_mean"], os.path.join(args.outdir, "subset2_distmult_mrr_sorted.png"))
    plot_relation_hist(metrics["test_mrr_by_relation_mean"], os.path.join(args.outdir, "subset2_distmult_mrr_hist.png"))
    write_summary(metrics, os.path.join(args.outdir, "subset2_distmult_summary.txt"))

    print(f"[ok] wrote distmult visualizations to: {args.outdir}")


if __name__ == "__main__":
    main()
