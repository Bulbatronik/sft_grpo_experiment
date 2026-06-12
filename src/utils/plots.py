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
        "diverse_10pct":  "#e41a1c",
        "random_10pct":   "#377eb8",
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
    """Visualise rollout difficulty and what each GRPO selection picked.

    With n binary rollouts the (mean, std) pairs take only n+1 discrete
    values, so a raw scatter collapses onto a handful of overplotted dots.
    Instead:
      left  — bar chart of the share of each set (pool + selections) falling
              in each mean-reward bin: shows *where* each selection draws from.
      right — pool count at each discrete (mean, std) point, bubble-sized,
              tracing the binomial std curve that variance selection ranks by.
    """
    sel_colors = {
        "variance_10pct": "#e41a1c",   # reds = variance
        "variance_20pct": "#ff7f7f",
        "random_10pct":   "#377eb8",   # blues = random
        "random_20pct":   "#7fb3d9",
    }

    levels = np.unique(np.round(mean_rewards, 6))
    n_lv = len(levels)

    def _shares(values: np.ndarray) -> np.ndarray:
        counts = np.array([(np.isclose(values, lv)).sum() for lv in levels], dtype=float)
        return counts / max(len(values), 1)

    fig, (ax_bar, ax_std) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]}
    )

    # ── Left: share of each set per mean-reward level ─────────────────────────
    groups = [("pool", None)] + list(selection_masks.items())
    n_groups = len(groups)
    width = 0.8 / n_groups
    x = np.arange(n_lv)

    for gi, (name, mask) in enumerate(groups):
        vals = mean_rewards if mask is None else mean_rewards[mask]
        color = "grey" if mask is None else sel_colors.get(name, "#444444")
        n = len(vals)
        ax_bar.bar(
            x + (gi - n_groups / 2 + 0.5) * width,
            _shares(vals),
            width=width * 0.95,
            color=color,
            label=f"{name} (n={n})",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{lv:.1f}" for lv in levels])
    ax_bar.set_xlabel("mean reward over rollouts (pass rate)")
    ax_bar.set_ylabel("share of set")
    ax_bar.set_title("Where each selection draws from")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis="y", linestyle=":", alpha=0.4)

    # ── Right: pool counts on the (mean, std) plane ───────────────────────────
    pts: dict[tuple[float, float], int] = {}
    for m, s in zip(np.round(mean_rewards, 6), np.round(std_rewards, 6)):
        pts[(m, s)] = pts.get((m, s), 0) + 1
    max_count = max(pts.values())
    for (m, s), c in sorted(pts.items()):
        ax_std.scatter([m], [s], s=80 + 2200 * c / max_count,
                       color="grey", alpha=0.45, edgecolor="black", zorder=3)
        ax_std.annotate(str(c), (m, s), ha="center", va="center",
                        fontsize=7.5, zorder=4)

    ax_std.set_xlabel("mean reward")
    ax_std.set_ylabel("std reward (variance-selection score)")
    ax_std.set_title("Pool difficulty profile")
    ax_std.set_xlim(-0.12, 1.12)
    ax_std.set_ylim(-0.06, max(std_rewards) * 1.25 + 0.01)
    ax_std.grid(True, linestyle=":", alpha=0.4)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_sft_loss_plot(
    train_curves: dict[str, list[tuple[int, float]]],
    val_curves: dict[str, list[tuple[int, float]]],
    out_path: str | Path,
) -> None:
    """Plot train and val loss curves per selection, x-axis = training step."""
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    all_names = sorted(set(list(train_curves) + list(val_curves)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(all_names):
        c = colors[i % len(colors)]
        if name in train_curves:
            steps, losses = zip(*train_curves[name])
            ax.plot(steps, losses, color=c, linestyle="-", label=f"{name} train")
        if name in val_curves:
            steps, losses = zip(*val_curves[name])
            ax.plot(steps, losses, color=c, linestyle="--", marker="o", markersize=4,
                    label=f"{name} val")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT train / val loss (solid=train, dashed=val)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_sft_accuracy_plot(
    acc_curves: dict[str, list[tuple[int, float]]],
    out_path: str | Path,
) -> None:
    """Plot validation accuracy curves per SFT selection, x-axis = training step."""
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    names = sorted(acc_curves)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        c = colors[i % len(colors)]
        steps, accs = zip(*acc_curves[name])
        ax.plot(
            steps, accs,
            color=c, linestyle="-", marker="o", markersize=5,
            label=name,
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("SFT validation accuracy (greedy, exact-match)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
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
