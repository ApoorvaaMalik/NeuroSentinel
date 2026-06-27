"""
visualizer.py — Publication-Quality Plots for GradSentinel
============================================================
KEY FIXES vs. original:
  - generate_all_plots now accepts task_boundaries from trainer history and
    passes them to plot_layer_risk_timeline (they were always None before).
  - plot_accuracy_vs_risk now takes the scorer's accuracy_history dict
    (list of {step, accuracy}) instead of the raw task_accuracies dict
    (which is only indexed by task name, not by step).
  - plot_forgetting_summary: guard against None values in task_accuracies.
  - Added plot_h1_scatter: risk vs. accuracy-drop scatter (supports H1 paper figure).
  - Added plot_h3_lag_bars: per-task/layer lag bars (supports H3 paper figure).
  - All plt.style references updated for modern matplotlib.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from collections import defaultdict

from risk_scorer import RISK_THRESHOLDS

LAYER_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#264653", "#8338EC"]
RISK_COLORS  = {"safe": "#2A9D8F", "warning": "#E9C46A", "danger": "#E63946"}


def _style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "legend.fontsize": 9, "figure.dpi": 150,
    })


class ForgettingVisualizer:

    def __init__(self, output_dir: str = "./results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        _style()
        print(f"[Visualizer] Plots → {output_dir}")

    # ─── PLOT 1: Per-Layer Risk Timeline ────────────────────────────────────

    def plot_layer_risk_timeline(self, risk_history: dict,
                                 task_boundaries: list = None):
        """Figure 1: per-layer risk over training steps with task dividers."""
        if not risk_history:
            print("[Visualizer] No risk history."); return None

        n = len(risk_history)
        fig, axes = plt.subplots(n, 1, figsize=(13, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, (layer, hist), color in zip(axes, risk_history.items(), LAYER_COLORS):
            if not hist:
                continue
            steps  = [e["step"] for e in hist]
            risks  = [e["risk"] for e in hist]
            w      = min(10, len(risks))
            smooth = np.convolve(risks, np.ones(w) / w, mode="same")

            ax.scatter(steps, risks,  alpha=0.15, color=color, s=8)
            ax.plot(steps, smooth,    color=color, linewidth=2, label="Smoothed risk")
            ax.axhline(RISK_THRESHOLDS["warning"], color="orange", ls="--",
                       alpha=0.7, lw=1.2, label=f"Warning ({RISK_THRESHOLDS['warning']})")
            ax.axhline(RISK_THRESHOLDS["danger"],  color="red",    ls="--",
                       alpha=0.7, lw=1.2, label=f"Danger  ({RISK_THRESHOLDS['danger']})")
            ax.fill_between(steps, RISK_THRESHOLDS["danger"],  1.0, alpha=0.07, color="red")
            ax.fill_between(steps, RISK_THRESHOLDS["warning"],
                            RISK_THRESHOLDS["danger"], alpha=0.07, color="orange")

            if task_boundaries:
                for i, b in enumerate(task_boundaries):
                    ax.axvline(b, color="purple", ls=":", lw=1.4, alpha=0.8)
                    ax.text(b + 1, 0.97, f"T{i}", rotation=90,
                            va="top", fontsize=7, color="purple")

            ax.set_ylabel("Forgetting Risk")
            ax.set_title(f"Layer: {layer}", fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="upper left", fontsize=8)

        axes[-1].set_xlabel("Global Training Step")
        fig.suptitle("GradSentinel — Per-Layer Forgetting Risk", fontsize=15,
                     fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "layer_risk_timeline.png")

    # ─── PLOT 2: Accuracy vs. Risk (H3) ─────────────────────────────────────

    def plot_accuracy_vs_risk(self, accuracy_during_training: dict,
                              risk_history: dict,
                              task_to_watch: str = None,
                              layer_to_watch: str = None):
        """
        Figure 2: dual-axis plot.
        Left  (blue):  accuracy on an old task, measured step-by-step during
                       a subsequent task's training.
        Right (red):   risk score for a chosen layer.

        FIX: `accuracy_during_training` must be scorer.accuracy_history, i.e.
        {task_name: [{step, accuracy}, ...]} — NOT trainer.history["task_accuracies"].
        """
        fig, ax1 = plt.subplots(figsize=(13, 5))
        ax2 = ax1.twinx()

        if task_to_watch and task_to_watch in accuracy_during_training:
            data       = accuracy_during_training[task_to_watch]
            acc_steps  = [e["step"]     for e in data]
            acc_vals   = [e["accuracy"] for e in data]
            ax1.plot(acc_steps, acc_vals, "b-o", lw=2, ms=4,
                     label=f"Accuracy on {task_to_watch}")
            ax1.set_ylabel("Test Accuracy (previous task)", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.set_ylim(-0.05, 1.15)

        if layer_to_watch and layer_to_watch in risk_history:
            data        = risk_history[layer_to_watch]
            risk_steps  = [e["step"] for e in data]
            risk_vals   = [e["risk"] for e in data]
            w           = min(10, len(risk_vals))
            smooth      = np.convolve(risk_vals, np.ones(w) / w, mode="same")
            ax2.plot(risk_steps, smooth, "r-", lw=2, alpha=0.85,
                     label=f"Risk [{layer_to_watch}]")
            ax2.axhline(RISK_THRESHOLDS["warning"], color="orange", ls="--", alpha=0.5)
            ax2.set_ylabel("Forgetting Risk Score", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(-0.05, 1.15)

        ax1.set_xlabel("Global Training Step")
        ax1.set_title("H3: Risk Spike Precedes Accuracy Drop\n"
                      "(risk line should spike BEFORE blue accuracy line falls)",
                      fontsize=12)
        l1, n1 = ax1.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, n1 + n2, loc="center left")
        plt.tight_layout()
        return self._save(fig, "accuracy_vs_risk.png")

    # ─── PLOT 3: Layer Risk Heatmap (H2) ────────────────────────────────────

    def plot_layer_risk_heatmap(self, risk_history: dict):
        """Figure 3: 2D heatmap (layer × step) coloured by risk score."""
        if not risk_history:
            return None

        layers    = list(risk_history.keys())
        all_steps = sorted({e["step"] for h in risk_history.values() for e in h})
        if not all_steps:
            return None

        step_idx = {s: i for i, s in enumerate(all_steps)}
        mat      = np.zeros((len(layers), len(all_steps)))

        for li, (layer, hist) in enumerate(risk_history.items()):
            for e in hist:
                mat[li, step_idx[e["step"]]] = e["risk"]
            # Smooth each row
            w = min(10, mat.shape[1])
            mat[li] = np.convolve(mat[li], np.ones(w) / w, mode="same")

        fig, ax = plt.subplots(figsize=(14, max(4, len(layers) * 1.3)))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                       interpolation="bilinear")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_ylabel("Layer (shallow → deep)")

        n_ticks = min(10, len(all_steps))
        tpos    = np.linspace(0, len(all_steps) - 1, n_ticks, dtype=int)
        ax.set_xticks(tpos)
        ax.set_xticklabels([str(all_steps[i]) for i in tpos], rotation=45)
        ax.set_xlabel("Global Training Step")

        plt.colorbar(im, ax=ax, label="Forgetting Risk")
        ax.set_title("H2: Layer-Wise Forgetting Risk Heatmap\n"
                     "(deeper layers expected to be brighter / higher risk)", fontsize=12)
        plt.tight_layout()
        return self._save(fig, "layer_risk_heatmap.png")

    # ─── PLOT 4: Forgetting Summary ──────────────────────────────────────────

    def plot_forgetting_summary(self, task_accuracies: dict):
        """Figure 4: grouped bar chart of accuracy retention per task."""
        if not task_accuracies:
            return None

        tasks   = list(task_accuracies.keys())
        n       = len(tasks)
        fig, ax = plt.subplots(figsize=(10, 5))
        width   = 0.8 / n
        x       = np.arange(n)

        for i, eval_task in enumerate(tasks):
            accs  = []
            for after_task in tasks:
                val = task_accuracies.get(after_task, {}).get(eval_task, None)
                accs.append(float(val) if val is not None else 0.0)

            bars = ax.bar(x + i * width, accs, width,
                          label=eval_task, color=LAYER_COLORS[i % len(LAYER_COLORS)],
                          alpha=0.82)
            for bar, val in zip(bars, accs):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(0.7, color="red", ls="--", alpha=0.5, label="0.70 threshold")
        ax.set_xticks(x + width * (n - 1) / 2)
        ax.set_xticklabels([f"After\n{t}" for t in tasks])
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(0, 1.18)
        ax.set_title("Catastrophic Forgetting — Accuracy After Each Task")
        ax.legend(fontsize=9)
        plt.tight_layout()
        return self._save(fig, "forgetting_summary.png")

    # ─── PLOT 5 (new): H1 Scatter ────────────────────────────────────────────

    def plot_h1_scatter(self, risk_vals: list, drop_vals: list,
                        pearson_r: float = None):
        """
        NEW Figure 5: scatter of (mean_risk, accuracy_drop) per task-transition.
        Helps reviewers immediately see H1 correlation.
        """
        if not risk_vals or not drop_vals:
            return None

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(risk_vals, drop_vals, color="#E63946", s=80, zorder=3)

        m, b = np.polyfit(risk_vals, drop_vals, 1)
        xs   = np.linspace(min(risk_vals), max(risk_vals), 100)
        ax.plot(xs, m * xs + b, "k--", lw=1.5, alpha=0.7,
                label=f"Linear fit  r={pearson_r:.2f}" if pearson_r else "Linear fit")

        ax.set_xlabel("Mean Gradient Conflict Risk")
        ax.set_ylabel("Accuracy Drop on Previous Task")
        ax.set_title("H1: Gradient Conflict vs. Accuracy Drop")
        ax.legend()
        plt.tight_layout()
        return self._save(fig, "h1_scatter.png")

    # ─── PLOT 6 (new): H3 Lag Bars ───────────────────────────────────────────

    def plot_h3_lag_bars(self, lag_results: list):
        """
        NEW Figure 6: bar chart showing early-warning lag per (task, layer) pair.
        Positive = warned early (green), negative = warned late (red).
        """
        if not lag_results:
            return None

        labels = [f"{r['task']}\n{r['layer']}" for r in lag_results
                  if r.get("status") == "computed"]
        lags   = [r["lag_steps"] for r in lag_results if r.get("status") == "computed"]

        if not lags:
            return None

        colors = ["#2A9D8F" if l > 0 else "#E63946" for l in lags]
        fig, ax = plt.subplots(figsize=(max(6, len(lags) * 1.2), 4))
        ax.bar(range(len(lags)), lags, color=colors, edgecolor="white", alpha=0.9)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Early-warning lag (steps)\nPositive = warned early OK")
        ax.set_title("H3: Early-Warning Lag per Task × Layer")
        plt.tight_layout()
        return self._save(fig, "h3_lag_bars.png")

    # ─── GENERATE ALL ────────────────────────────────────────────────────────

    def generate_all_plots(self, experiment_results: dict,
                           risk_history: dict,
                           scorer=None):
        """
        Call all plot functions.

        Parameters
        ----------
        experiment_results : dict  — trainer.history
        risk_history       : dict  — {layer: [{step, risk, ...}]}
        scorer             : RiskScorer | None — if provided, H1/H3 plots are made
        """
        print("\n[Visualizer] Generating plots...")
        paths = []

        boundaries = experiment_results.get("task_boundaries", [])
        paths.append(self.plot_layer_risk_timeline(risk_history, boundaries))

        # H3 dual-axis plot — use scorer's accuracy_history if available
        if scorer and scorer.accuracy_history and risk_history:
            first_task  = list(scorer.accuracy_history.keys())[0]
            first_layer = list(risk_history.keys())[-1]  # deepest layer
            paths.append(self.plot_accuracy_vs_risk(
                accuracy_during_training=scorer.accuracy_history,
                risk_history=risk_history,
                task_to_watch=first_task,
                layer_to_watch=first_layer,
            ))
        else:
            print("[Visualizer] Skipping H3 plot — no in-training accuracy data.")

        paths.append(self.plot_layer_risk_heatmap(risk_history))
        paths.append(self.plot_forgetting_summary(
            experiment_results.get("task_accuracies", {})))

        # H1 scatter
        if scorer:
            h1 = scorer._test_H1()
            if "pearson_r" in h1:
                # Reconstruct the paired data for the scatter
                # (stored internally; here we just re-derive it)
                task_names = list(scorer.accuracy_history.keys())
                rv, dv = [], []
                for tn in task_names:
                    acc_data = scorer.accuracy_history[tn]
                    if len(acc_data) < 2: continue
                    drop = acc_data[0]["accuracy"] - acc_data[-1]["accuracy"]
                    start, end = acc_data[0]["step"], acc_data[-1]["step"]
                    lr_list = []
                    for layer, hist in scorer.layer_risk_history.items():
                        lr_list += [e["risk"] for e in hist
                                    if start <= e["step"] <= end]
                    if lr_list:
                        rv.append(float(np.mean(lr_list)))
                        dv.append(float(drop))
                paths.append(self.plot_h1_scatter(rv, dv, h1.get("pearson_r")))

        # H3 lag bars
        if scorer:
            lag_results = []
            for tn in scorer.accuracy_history:
                for layer in scorer.layer_risk_history:
                    lag_results.append(
                        scorer.compute_early_warning_lag(tn, layer))
            paths.append(self.plot_h3_lag_bars(lag_results))

        valid = [p for p in paths if p]
        print(f"[Visualizer] {len(valid)} plots saved:")
        for p in valid:
            print(f"  {p}")
        return valid

    # ─── HELPER ─────────────────────────────────────────────────────────────

    def _save(self, fig, name: str) -> str:
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer] Saved: {path}")
        return path