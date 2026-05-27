"""Reusable plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_pca_variance_plot(
    explained_variance_ratio: np.ndarray,
    out_path: str | Path,
    variance_threshold: float = 0.95,
) -> None:
    cumvar = np.cumsum(explained_variance_ratio)
    n_keep = int(np.searchsorted(cumvar, variance_threshold) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Per-component variance")

    axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, marker=".")
    axes[1].axhline(variance_threshold * 100, color="red", linestyle="--", label=f"{variance_threshold*100:.0f}% threshold")
    axes[1].axvline(n_keep, color="orange", linestyle="--", label=f"n_keep={n_keep}")
    axes[1].set_xlabel("Components")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative explained variance")
    axes[1].legend()

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_sft_selection_pca_plot(
    reduced: np.ndarray,
    selections: dict[str, np.ndarray],
    out_dir: str | Path,
) -> None:
    """Save one PC1×PC2 scatter per selection, each highlighted against the full pool."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "diverse_5pct":  "#e41a1c",
        "random_5pct":   "#377eb8",
        "diverse_20pct": "#4daf4a",
        "random_20pct":  "#984ea3",
    }

    for name, idxs in selections.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(reduced[:, 0], reduced[:, 1], c="lightgrey", s=4, alpha=0.4, label="all", zorder=1)
        ax.scatter(
            reduced[idxs, 0],
            reduced[idxs, 1],
            c=colors.get(name, "#ff7f00"),
            s=18,
            alpha=0.8,
            label=f"{name} (n={len(idxs)})",
            zorder=2,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"SFT selection — {name}")
        ax.legend(markerscale=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"sft_selection_pca_{name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def save_grpo_reward_scatter(
    mean_rewards: np.ndarray,
    std_rewards: np.ndarray,
    selection_masks: dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "",
) -> None:
    """Scatter mean_reward vs std_reward with marginal histograms."""
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    colors = {"pool": "lightgrey", "variance": "#e41a1c", "random": "#377eb8"}

    ax_main.scatter(mean_rewards, std_rewards, c="lightgrey", s=4, alpha=0.3, label="pool")
    for name, mask in selection_masks.items():
        c = colors.get(name, "#444444")
        ax_main.scatter(mean_rewards[mask], std_rewards[mask], c=c, s=16, alpha=0.7, label=name)

    ax_main.set_xlabel("mean reward")
    ax_main.set_ylabel("std reward")
    if title:
        ax_main.set_title(title)
    ax_main.legend(fontsize=8)

    ax_top.hist(mean_rewards, bins=40, color="lightgrey", edgecolor="none")
    ax_top.set_ylabel("count")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_right.hist(std_rewards, bins=40, color="lightgrey", edgecolor="none", orientation="horizontal")
    ax_right.set_xlabel("count")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_sft_loss_plot(
    loss_curves: dict[str, list[float]],
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, losses in loss_curves.items():
        ax.plot(losses, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT training loss comparison")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_final_accuracy_bar(
    results: dict,
    base_acc: float,
    sft_only_accs: dict[str, float],
    out_path: str | Path,
) -> None:
    """Grouped bar chart: SFT strategy groups, GRPO strategy as inner grouping."""
    sft_keys = sorted({k[0] for k in results})
    grpo_keys = sorted({k[1] for k in results})
    n_sft = len(sft_keys)
    n_grpo = len(grpo_keys)
    width = 0.7 / n_grpo
    x = np.arange(n_sft)

    fig, ax = plt.subplots(figsize=(max(10, n_sft * 3), 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for gi, gkey in enumerate(grpo_keys):
        vals = [results.get((sk, gkey), float("nan")) for sk in sft_keys]
        ax.bar(x + gi * width, vals, width, label=gkey, color=colors[gi % len(colors)])

    ax.axhline(base_acc, color="black", linestyle="--", linewidth=1.5, label=f"base ({base_acc:.3f})")
    for sname, sacc in sft_only_accs.items():
        ax.axhline(sacc, linestyle=":", linewidth=1, label=f"sft-only {sname} ({sacc:.3f})")

    ax.set_xticks(x + width * (n_grpo - 1) / 2)
    ax.set_xticklabels(sft_keys, rotation=15, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Final GSM8K test accuracy by selection strategy")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
