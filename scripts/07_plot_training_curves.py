#!/usr/bin/env python3
"""
Phase 7 — Plot accuracy training curves as a branching tree.

X-axis 0–100:
  0–40   SFT eval steps (logged every 10 steps)
  50     Branch point — SFT best-checkpoint accuracy (what GRPO starts from)
  60–100 GRPO eval steps (GRPO step 10→x=60 … step 50→x=100)

Outputs:
  results/plots/curves_<sft_sel>.png  — 1 plot per SFT selection (SFT + 4 GRPO)
  results/plots/curves_all.png        — combined plot (all 4 SFT + 16 GRPO lines)
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

SFT_SELECTIONS  = ["diverse_5pct", "random_5pct", "diverse_20pct", "random_20pct"]
GRPO_SELECTIONS = ["variance_5pct", "random_5pct", "variance_20pct", "random_20pct"]

_DEFAULT_LOGS_DIR    = ROOT / "logs"
_DEFAULT_RESULTS_DIR = ROOT / "results"

# Array task index → (SFT sel, GRPO sel)
TASK_MAP = {
    i: (SFT_SELECTIONS[i // 4], GRPO_SELECTIONS[i % 4])
    for i in range(16)
}


def _latest_log(logs_dir: Path, pattern: str) -> Path | None:
    """Return the most recently modified file matching a glob pattern in logs_dir."""
    candidates = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _detect_grpo_job(logs_dir: Path) -> str | None:
    """Find the job ID of the most recent GRPO array run by looking at task-0 logs."""
    log = _latest_log(logs_dir, "gsm8k_grpo-*_0.out")
    if log is None:
        return None
    # filename: gsm8k_grpo-<job_id>_0.out
    return log.stem.split("-")[1].split("_")[0]

# Colors for the 4 SFT selections
SFT_COLORS = {
    "diverse_5pct":  "#2196F3",   # blue
    "random_5pct":   "#F44336",   # red
    "diverse_20pct": "#4CAF50",   # green
    "random_20pct":  "#FF9800",   # orange
}

# Line styles for the 4 GRPO selections
GRPO_STYLES = {
    "variance_5pct":  "-",
    "random_5pct":    "--",
    "variance_20pct": "-.",
    "random_20pct":   ":",
}

GRPO_MARKERS = {
    "variance_5pct":  "o",
    "random_5pct":    "s",
    "variance_20pct": "^",
    "random_20pct":   "D",
}


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_sft_log(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Returns {sft_sel: [(step, accuracy), ...]}."""
    results: dict[str, list] = {sel: [] for sel in SFT_SELECTIONS}
    current_sel = None

    sel_re  = re.compile(r"=== Starting SFT run: (\S+) ===")
    eval_re = re.compile(r"step:(\d+) - eval/acc:([\d.]+)")
    best_re = re.compile(r"\[(\S+)\] Best step: (\d+)")

    best_steps: dict[str, int] = {}

    with open(log_path) as f:
        for line in f:
            m = sel_re.search(line)
            if m:
                current_sel = m.group(1)
                continue
            m = eval_re.search(line)
            if m and current_sel:
                results[current_sel].append((int(m.group(1)), float(m.group(2))))
                continue
            m = best_re.search(line)
            if m:
                best_steps[m.group(1)] = int(m.group(2))

    return results, best_steps


def parse_grpo_log(log_path: Path) -> list[tuple[int, float]]:
    """Returns [(global_step, val_accuracy), ...] from one GRPO task log."""
    points = []
    pattern = re.compile(
        r"val-core/gsm8k/acc/mean@1:([\d.]+).*?training/global_step:(\d+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                acc  = float(m.group(1))
                step = int(m.group(2))
                points.append((step, acc))
    return points


def load_all_grpo(job_id: str, logs_dir: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """Returns {(sft_sel, grpo_sel): [(step, acc), ...]}."""
    data = {}
    for task_idx, (sft_sel, grpo_sel) in TASK_MAP.items():
        log_path = logs_dir / f"gsm8k_grpo-{job_id}_{task_idx}.out"
        if not log_path.exists():
            print(f"WARNING: {log_path} not found")
            continue
        points = parse_grpo_log(log_path)
        data[(sft_sel, grpo_sel)] = points
    return data


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _sft_x_offset(step: int) -> int:
    """SFT step → x-axis position (identity: 0,10,20,30,40)."""
    return step


def _grpo_x_offset(step: int) -> int:
    """GRPO step → x-axis position (shift +50: 10→60, …, 50→100)."""
    return step + 50


def _draw_phase_separator(ax):
    ax.axvline(x=50, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(50, ax.get_ylim()[0] + 0.005, " SFT→GRPO", fontsize=7,
            color="grey", va="bottom")


def _set_axes(ax, title=""):
    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel("GSM8K accuracy (val)", fontsize=10)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xticklabels(
        ["0", "10", "20", "30", "40", "best*", "60", "70", "80", "90", "100"],
        fontsize=8,
    )
    ax.set_xlim(-3, 103)
    ax.grid(True, linestyle=":", alpha=0.4)
    if title:
        ax.set_title(title, fontsize=11)


# ── Per-SFT-selection plots ───────────────────────────────────────────────────

def plot_per_selection(
    sft_data: dict,
    best_steps: dict,
    grpo_data: dict,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for sft_sel in SFT_SELECTIONS:
        fig, ax = plt.subplots(figsize=(9, 5))
        color = SFT_COLORS[sft_sel]

        # SFT curve
        pts = sft_data[sft_sel]
        xs_sft = [_sft_x_offset(s) for s, _ in pts]
        ys_sft = [a for _, a in pts]
        ax.plot(xs_sft, ys_sft, color=color, linewidth=2.2,
                marker="o", markersize=5, label=f"SFT {sft_sel}", zorder=3)

        # Branch point at x=50: accuracy of the best SFT checkpoint
        best_step = best_steps.get(sft_sel, 10)
        best_acc  = dict(pts).get(best_step, ys_sft[-1])
        ax.scatter([50], [best_acc], color=color, marker="*", s=160,
                   zorder=5, label=f"SFT best (step {best_step})")
        # Dotted bridge from last SFT point to branch point
        if xs_sft:
            ax.plot([xs_sft[-1], 50], [ys_sft[-1], best_acc],
                    color=color, linestyle=":", linewidth=1.2, alpha=0.6)

        # GRPO curves
        for grpo_sel in GRPO_SELECTIONS:
            pts_g = grpo_data.get((sft_sel, grpo_sel), [])
            if not pts_g:
                continue
            xs_g = [50] + [_grpo_x_offset(s) for s, _ in pts_g]
            ys_g = [best_acc] + [a for _, a in pts_g]
            ax.plot(
                xs_g, ys_g,
                color=color,
                linestyle=GRPO_STYLES[grpo_sel],
                marker=GRPO_MARKERS[grpo_sel],
                markersize=5,
                linewidth=1.6,
                alpha=0.85,
                label=f"GRPO {grpo_sel}",
            )

        _set_axes(ax, title=f"SFT: {sft_sel} → 4 GRPO branches")
        _draw_phase_separator(ax)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        out_path = out_dir / f"curves_{sft_sel}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Combined plot ─────────────────────────────────────────────────────────────

def plot_combined(
    sft_data: dict,
    best_steps: dict,
    grpo_data: dict,
    out_dir: Path,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    for sft_sel in SFT_SELECTIONS:
        color = SFT_COLORS[sft_sel]
        pts   = sft_data[sft_sel]
        xs_sft = [_sft_x_offset(s) for s, _ in pts]
        ys_sft = [a for _, a in pts]

        # Thick SFT line
        ax.plot(xs_sft, ys_sft, color=color, linewidth=2.5,
                marker="o", markersize=6, zorder=3)

        # Branch point at x=50
        best_step = best_steps.get(sft_sel, 10)
        best_acc  = dict(pts).get(best_step, ys_sft[-1])
        ax.scatter([50], [best_acc], color=color, marker="*", s=180, zorder=5)
        if xs_sft:
            ax.plot([xs_sft[-1], 50], [ys_sft[-1], best_acc],
                    color=color, linestyle=":", linewidth=1.0, alpha=0.5)

        # Thin GRPO lines
        for grpo_sel in GRPO_SELECTIONS:
            pts_g = grpo_data.get((sft_sel, grpo_sel), [])
            if not pts_g:
                continue
            xs_g = [50] + [_grpo_x_offset(s) for s, _ in pts_g]
            ys_g = [best_acc] + [a for _, a in pts_g]
            ax.plot(
                xs_g, ys_g,
                color=color,
                linestyle=GRPO_STYLES[grpo_sel],
                marker=GRPO_MARKERS[grpo_sel],
                markersize=4,
                linewidth=1.3,
                alpha=0.75,
            )

    _set_axes(ax, title="GSM8K accuracy: SFT → GRPO training trajectories")
    _draw_phase_separator(ax)

    # Legend: SFT colors + GRPO styles
    sft_handles = [
        mlines.Line2D([], [], color=SFT_COLORS[s], linewidth=2.5,
                      marker="o", markersize=6, label=f"SFT {s}")
        for s in SFT_SELECTIONS
    ]
    grpo_handles = [
        mlines.Line2D([], [], color="grey",
                      linestyle=GRPO_STYLES[g],
                      marker=GRPO_MARKERS[g],
                      markersize=5, linewidth=1.3,
                      label=f"GRPO {g}")
        for g in GRPO_SELECTIONS
    ]
    ax.legend(handles=sft_handles + grpo_handles,
              fontsize=8, ncol=2, loc="lower right")

    ax.text(0.01, 0.97,
            "★ = SFT best checkpoint (GRPO start)\n"
            "color = SFT selection   style = GRPO selection",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.tight_layout()
    out_path = out_dir / "curves_all.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Plot SFT→GRPO accuracy training curves.")
    p.add_argument("--logs-dir", default=str(_DEFAULT_LOGS_DIR),
                   help="Directory containing SLURM log files (default: logs/)")
    p.add_argument("--results-dir", default=str(_DEFAULT_RESULTS_DIR),
                   help="Root results directory; plots go to <results-dir>/plots/ (default: results/)")
    args = p.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir  = Path(args.results_dir) / "plots"

    sft_log = _latest_log(logs_dir, "gsm8k_sft-*.out")
    if sft_log is None:
        print("ERROR: no gsm8k_sft-*.out log found in", logs_dir)
        sys.exit(1)
    print(f"SFT log:  {sft_log.name}")

    grpo_job = _detect_grpo_job(logs_dir)
    if grpo_job is None:
        print("ERROR: no gsm8k_grpo-*_0.out log found in", logs_dir)
        sys.exit(1)
    print(f"GRPO job: {grpo_job}")

    print("Parsing SFT log …")
    sft_data, best_steps = parse_sft_log(sft_log)
    for sel, pts in sft_data.items():
        print(f"  {sel}: {len(pts)} eval points, best step = {best_steps.get(sel)}")

    print("Parsing GRPO logs …")
    grpo_data = load_all_grpo(grpo_job, logs_dir)
    print(f"  Loaded {len(grpo_data)} GRPO runs")

    print("Plotting per-selection …")
    plot_per_selection(sft_data, best_steps, grpo_data, out_dir)

    print("Plotting combined …")
    plot_combined(sft_data, best_steps, grpo_data, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
