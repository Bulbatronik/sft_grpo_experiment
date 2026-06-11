#!/usr/bin/env python3
"""
Phase 7 — Plot accuracy training curves as a branching tree.

X-axis shows real training steps:
  0 … best_step       SFT eval points (every 10 steps)
  best_step           Branch point — SFT best-checkpoint accuracy
  best_step … +300    GRPO eval points (every 20 steps, offset by best_step)

The branch point is selection-specific: each SFT selection may converge at a
different step, so each GRPO family starts at its own x position.

Outputs:
  results/plots/curves_<sft_sel>.png  — 1 plot per SFT selection (SFT + 4 GRPO branches)
  results/plots/curves_all.png        — combined plot (all 4 SFT + 16 GRPO lines)
  results/sft_eval_metrics.csv        — SFT eval accuracy per step (from logs)
  results/grpo_metrics.csv            — GRPO per-step metrics (reward, losses, val acc)
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

SFT_SELECTIONS  = ["diverse_10pct", "random_10pct", "diverse_20pct", "random_20pct"]
GRPO_SELECTIONS = ["variance_10pct", "random_10pct", "variance_20pct", "random_20pct"]

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


def _detect_array_job(logs_dir: Path, prefix: str) -> str | None:
    """Find the job ID of the most recent array run by looking at task-0 logs."""
    log = _latest_log(logs_dir, f"{prefix}-*_0.out")
    if log is None:
        return None
    # filename: <prefix>-<job_id>_0.out
    return log.stem.split("-")[1].split("_")[0]


# Colors for the 4 SFT selections
SFT_COLORS = {
    "diverse_10pct":  "#2196F3",   # blue
    "random_10pct":   "#F44336",   # red
    "diverse_20pct":  "#4CAF50",   # green
    "random_20pct":   "#FF9800",   # orange
}

# Line styles for the 4 GRPO selections
GRPO_STYLES = {
    "variance_10pct":  "-",
    "random_10pct":    "--",
    "variance_20pct":  "-.",
    "random_20pct":    ":",
}

GRPO_MARKERS = {
    "variance_10pct":  "o",
    "random_10pct":    "s",
    "variance_20pct":  "^",
    "random_20pct":    "D",
}


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_sft_logs(log_paths: list[Path]) -> tuple[dict, dict]:
    """
    Returns ({sft_sel: [(step, accuracy), ...]}, {sft_sel: best_step}).

    Accepts multiple log files because SFT runs as a SLURM array — one
    selection per task, each with its own log. Selections are identified by
    the "=== Starting SFT run: <sel> ===" marker inside each file, so the
    file→selection mapping doesn't matter.
    """
    results: dict[str, list] = {sel: [] for sel in SFT_SELECTIONS}
    best_steps: dict[str, int] = {}

    sel_re  = re.compile(r"=== Starting SFT run: (\S+) ===")
    eval_re = re.compile(r"step:(\d+) - eval/acc:([\d.]+)")
    best_re = re.compile(r"\[(\S+)\] Best step: (\d+)")

    for log_path in log_paths:
        current_sel = None
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


# verl console-logger metric names (one "step:N - k:v - k:v …" line per step).
# Val metrics appear on test_freq steps and on the standalone step-0 line when
# val_before_train is enabled; training metrics appear on every step line.
_GRPO_METRICS = {
    "val_acc":         re.compile(r"val-core/gsm8k/acc/mean@1:([\d.eE+\-]+)"),
    "reward_mean":     re.compile(r"critic/rewards/mean:([\d.eE+\-]+)"),
    "pg_loss":         re.compile(r"actor/pg_loss:([\d.eE+\-]+)"),
    "kl_loss":         re.compile(r"actor/kl_loss:([\d.eE+\-]+)"),
    "entropy":         re.compile(r"actor/entropy:([\d.eE+\-]+)"),
    "grad_norm":       re.compile(r"actor/grad_norm:([\d.eE+\-]+)"),
    "response_length": re.compile(r"response_length/mean:([\d.eE+\-]+)"),
}
# Leading step number; (?<![\w/]) excludes e.g. "timing_s/step:…".
_GRPO_STEP_RE = re.compile(r"(?<![\w/])step:(\d+) - ")


def parse_grpo_log(log_path: Path) -> list[dict]:
    """
    Returns one dict per logged step: {"step": int, <metric>: float|None …}
    for every metric in _GRPO_METRICS.
    """
    rows: dict[int, dict] = {}

    with open(log_path) as f:
        for line in f:
            m = _GRPO_STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            found = {k: pat.search(line) for k, pat in _GRPO_METRICS.items()}
            if not any(found.values()):
                continue
            row = rows.setdefault(
                step, {"step": step, **{k: None for k in _GRPO_METRICS}}
            )
            for key, mm in found.items():
                if mm:
                    row[key] = float(mm.group(1))

    return sorted(rows.values(), key=lambda r: r["step"])


def load_all_grpo(
    job_id: str, logs_dir: Path
) -> tuple[dict[tuple[str, str], list[tuple[int, float]]], dict[tuple[str, str], list[dict]]]:
    """
    Returns:
        acc_data  — {(sft_sel, grpo_sel): [(step, val_acc), ...]}  (for plotting)
        full_data — {(sft_sel, grpo_sel): [row_dict, ...]}          (for CSV)
    """
    acc_data: dict[tuple[str, str], list[tuple[int, float]]] = {}
    full_data: dict[tuple[str, str], list[dict]] = {}
    for task_idx, (sft_sel, grpo_sel) in TASK_MAP.items():
        log_path = logs_dir / f"gsm8k_grpo-{job_id}_{task_idx}.out"
        if not log_path.exists():
            print(f"WARNING: {log_path} not found")
            continue
        rows = parse_grpo_log(log_path)
        full_data[(sft_sel, grpo_sel)] = rows
        acc_data[(sft_sel, grpo_sel)] = [
            (r["step"], r["val_acc"]) for r in rows if r["val_acc"] is not None
        ]
    return acc_data, full_data


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _grpo_x(step: int, best_sft_step: int) -> int:
    """GRPO global_step → x-axis position (offset by the SFT branch step)."""
    return best_sft_step + step


def _compute_x_max(
    sft_data: dict,
    best_steps: dict,
    grpo_data: dict,
) -> int:
    all_x = [0]
    for sel, pts in sft_data.items():
        all_x.extend(s for s, _ in pts)
    for (sft_sel, _), pts in grpo_data.items():
        best = best_steps.get(sft_sel, 0)
        all_x.extend(best + s for s, _ in pts)
    return max(all_x) if all_x else 100


def _set_axes(ax, x_max: int, title: str = ""):
    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel("GSM8K accuracy (val)", fontsize=10)
    tick_int = 20 if x_max <= 300 else 50
    ax.set_xticks(range(0, x_max + tick_int, tick_int))
    ax.set_xlim(-5, x_max + tick_int // 2)
    ax.grid(True, linestyle=":", alpha=0.4)
    if title:
        ax.set_title(title, fontsize=11)


def _grpo_xs_ys(
    pts_g: list[tuple[int, float]],
    best_step: int,
    best_acc: float,
) -> tuple[list, list]:
    """Build GRPO x/y lists, anchoring at best_step if step-0 data is absent."""
    xs = [_grpo_x(s, best_step) for s, _ in pts_g]
    ys = [a for _, a in pts_g]
    if not pts_g or pts_g[0][0] > 0:
        xs = [best_step] + xs
        ys = [best_acc] + ys
    return xs, ys


# ── CSV export ───────────────────────────────────────────────────────────────

def save_grpo_csv(
    full_data: dict[tuple[str, str], list[dict]],
    best_steps: dict,
    out_dir: Path,
) -> None:
    """Save grpo_metrics.csv with one row per (sft_sel, grpo_sel, step)."""
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "grpo_metrics.csv"
    metric_keys = list(_GRPO_METRICS)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sft_sel", "grpo_sel", "grpo_step", "abs_step"] + metric_keys)
        for (sft_sel, grpo_sel), rows in sorted(full_data.items()):
            best = best_steps.get(sft_sel, 0)
            for r in rows:
                writer.writerow(
                    [sft_sel, grpo_sel, r["step"], best + r["step"]]
                    + [r[k] if r[k] is not None else "" for k in metric_keys]
                )
    print(f"Saved: {csv_path}")


def save_sft_csv(
    sft_data: dict,
    best_steps: dict,
    out_dir: Path,
) -> None:
    """
    Save sft_metrics.csv with accuracy per eval step (parsed from SLURM logs).
    Full training-loss data is richer in the sft_metrics.csv written by 02_train_sft.py;
    this file captures what's available from the log.
    """
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sft_eval_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sft_sel", "step", "val_acc", "is_best"])
        for sel, pts in sorted(sft_data.items()):
            best = best_steps.get(sel)
            for step, acc in pts:
                writer.writerow([sel, step, acc, int(step == best)])
    print(f"Saved: {csv_path}")


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

        pts = sft_data[sft_sel]
        xs_sft = [s for s, _ in pts]
        ys_sft = [a for _, a in pts]
        ax.plot(xs_sft, ys_sft, color=color, linewidth=2.2,
                marker="o", markersize=5, label=f"SFT {sft_sel}", zorder=3)

        best_step = best_steps.get(sft_sel, xs_sft[-1] if xs_sft else 0)
        best_acc  = dict(pts).get(best_step, ys_sft[-1] if ys_sft else 0.0)
        ax.scatter([best_step], [best_acc], color=color, marker="*", s=180,
                   zorder=5, label=f"SFT best (step {best_step})")

        # Vertical marker at branch point
        ax.axvline(x=best_step, color=color, linestyle="--",
                   linewidth=0.8, alpha=0.5)

        # GRPO curves
        for grpo_sel in GRPO_SELECTIONS:
            pts_g = grpo_data.get((sft_sel, grpo_sel), [])
            if not pts_g:
                continue
            xs_g, ys_g = _grpo_xs_ys(pts_g, best_step, best_acc)
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

        per_grpo = {k: v for k, v in grpo_data.items() if k[0] == sft_sel}
        x_max = _compute_x_max({sft_sel: pts}, best_steps, per_grpo)
        _set_axes(ax, x_max, title=f"SFT: {sft_sel} → 4 GRPO branches")
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
    max_sft_steps: int = 200,
    max_grpo_steps: int = 300,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    for sft_sel in SFT_SELECTIONS:
        color = SFT_COLORS[sft_sel]
        pts   = sft_data[sft_sel]
        xs_sft = [s for s, _ in pts]
        ys_sft = [a for _, a in pts]

        ax.plot(xs_sft, ys_sft, color=color, linewidth=2.5,
                marker="o", markersize=6, zorder=3)

        best_step = best_steps.get(sft_sel, xs_sft[-1] if xs_sft else 0)
        best_acc  = dict(pts).get(best_step, ys_sft[-1] if ys_sft else 0.0)
        ax.scatter([best_step], [best_acc], color=color, marker="*", s=200, zorder=5)

        for grpo_sel in GRPO_SELECTIONS:
            pts_g = grpo_data.get((sft_sel, grpo_sel), [])
            if not pts_g:
                continue
            xs_g, ys_g = _grpo_xs_ys(pts_g, best_step, best_acc)
            ax.plot(
                xs_g, ys_g,
                color=color,
                linestyle=GRPO_STYLES[grpo_sel],
                marker=GRPO_MARKERS[grpo_sel],
                markersize=4,
                linewidth=1.3,
                alpha=0.75,
            )

    # Shade the SFT→GRPO transition zone
    active_best = [v for v in best_steps.values() if v > 0]
    if active_best:
        lo, hi = min(active_best), max(active_best)
        ax.axvspan(lo - 5, hi + 5, alpha=0.07, color="grey", label="SFT→GRPO zone")

    # Fixed x-axis so all combined plots are comparable regardless of early stopping.
    # x_max = SFT upper bound + GRPO upper bound (worst case, no early stopping).
    x_max = max_sft_steps + max_grpo_steps
    _set_axes(ax, x_max, title="GSM8K accuracy: SFT → GRPO training trajectories")

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
    p.add_argument("--max-sft-steps", type=int, default=200,
                   help="Fixed upper bound for SFT training steps in the combined plot (default: 200)")
    p.add_argument("--max-grpo-steps", type=int, default=300,
                   help="Fixed upper bound for GRPO training steps in the combined plot (default: 300)")
    args = p.parse_args()

    logs_dir    = Path(args.logs_dir)
    results_dir = Path(args.results_dir)
    out_dir     = results_dir / "plots"

    # SFT runs as a SLURM array: one log per selection, all sharing a job ID.
    sft_job = _detect_array_job(logs_dir, "gsm8k_sft")
    if sft_job is not None:
        sft_logs = sorted(logs_dir.glob(f"gsm8k_sft-{sft_job}_*.out"))
    else:
        # Fall back to a single non-array log (e.g. `make sft` direct run).
        single = _latest_log(logs_dir, "gsm8k_sft-*.out")
        sft_logs = [single] if single else []
    if not sft_logs:
        print("ERROR: no gsm8k_sft-*.out log found in", logs_dir)
        sys.exit(1)
    print(f"SFT logs: {', '.join(p.name for p in sft_logs)}")

    grpo_job = _detect_array_job(logs_dir, "gsm8k_grpo")
    if grpo_job is None:
        print("ERROR: no gsm8k_grpo-*_0.out log found in", logs_dir)
        sys.exit(1)
    print(f"GRPO job: {grpo_job}")

    print("Parsing SFT logs …")
    sft_data, best_steps = parse_sft_logs(sft_logs)
    for sel, pts in sft_data.items():
        print(f"  {sel}: {len(pts)} eval points, best step = {best_steps.get(sel)}")

    print("Parsing GRPO logs …")
    grpo_data, grpo_full = load_all_grpo(grpo_job, logs_dir)
    print(f"  Loaded {len(grpo_data)} GRPO runs")

    print("Saving CSVs …")
    save_sft_csv(sft_data, best_steps, results_dir)
    save_grpo_csv(grpo_full, best_steps, results_dir)

    print("Plotting per-selection …")
    plot_per_selection(sft_data, best_steps, grpo_data, out_dir)

    print("Plotting combined …")
    plot_combined(sft_data, best_steps, grpo_data, out_dir,
                  max_sft_steps=args.max_sft_steps,
                  max_grpo_steps=args.max_grpo_steps)

    print("Done.")


if __name__ == "__main__":
    main()
