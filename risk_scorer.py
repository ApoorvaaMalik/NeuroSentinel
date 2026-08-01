"""
risk_scorer.py — Converting Raw Gradient Signals into Forgetting Risk Scores
=============================================================================

HYPOTHESES
----------
H1: Gradient conflict score is positively correlated with accuracy drop magnitude.
H2: Forgetting is non-uniform across layers (deeper layers have higher risk).
H3: GradSentinel detects risk N steps BEFORE accuracy drops (early warning).
"""

import numpy as np
from collections import defaultdict
from scipy import stats      # pip install scipy
import json
import os


# ─── SHARED THRESHOLDS 

RISK_THRESHOLDS = {
    "safe":    0.40,
    "warning": 0.65,
    "danger":  0.80,
}


def risk_label(score: float) -> str:
    if score >= RISK_THRESHOLDS["danger"]:
        return "DANGER"
    elif score >= RISK_THRESHOLDS["warning"]:
        return "WARNING"
    return "SAFE"


class RiskScorer:
    """
    Collects risk readings from the Auditor, applies smoothing, issues
    threshold warnings, and tests the three core hypotheses.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size

        # {layer_name: [{"step": int, "risk": float, "conflict": float, ...}, ...]}
        self.layer_risk_history = defaultdict(list)

        # {task_name: [{"step": int, "accuracy": float}, ...]}
        self.accuracy_history = defaultdict(list)

        # All warnings that were issued
        self.warnings_issued = []

    # ─── INGEST 
    def process_audit_snapshot(self, risk_data: list) -> list:
        """
        Ingest a list of per-layer risk dicts from auditor.get_risk_snapshot().
        Returns any warnings triggered by this snapshot.
        """
        warnings = []
        for entry in risk_data:
            layer = entry["layer"]
            self.layer_risk_history[layer].append({
                "step":          entry["step"],
                "risk":          entry["risk"],
                "conflict":      entry.get("conflict", 0.0),
                "fisher_weight": entry.get("fisher_weight", 0.0),
                "task_idx":      entry.get("task_idx", -1),
            })
            w = self._check_threshold(layer, entry["risk"], entry["step"])
            if w:
                warnings.append(w)
                self.warnings_issued.append(w)
        return warnings

    def record_accuracy(self, task_name: str, step: int, accuracy: float):
        """
        Called by the trainer whenever evaluation runs during training.
        Required for H1 correlation and H3 early-warning lag.
        """
        self.accuracy_history[task_name].append({"step": step, "accuracy": accuracy})

    # SMOOTHING & THRESHOLDS

    def get_smoothed_risk(self, layer: str) -> float:
        history = self.layer_risk_history.get(layer, [])
        if not history:
            return 0.0
        recent = history[-self.window_size:]
        return float(np.mean([e["risk"] for e in recent]))

    def get_risk_trend(self, layer: str) -> str:
        history = self.layer_risk_history.get(layer, [])
        if len(history) < 10:
            return "insufficient_data"
        recent = [e["risk"] for e in history[-self.window_size:]]
        mid    = len(recent) // 2
        delta  = np.mean(recent[mid:]) - np.mean(recent[:mid])
        if delta >  0.05: return "increasing"
        if delta < -0.05: return "decreasing"
        return "stable"

    def _check_threshold(self, layer: str, risk: float, step: int):
        smoothed = self.get_smoothed_risk(layer)
        if smoothed >= RISK_THRESHOLDS["danger"]:
            level   = "DANGER"
            message = f"🚨 [{layer}] very high forgetting risk ({smoothed:.2f})"
        elif smoothed >= RISK_THRESHOLDS["warning"]:
            level   = "WARNING"
            message = f"⚠️  [{layer}] elevated forgetting risk ({smoothed:.2f})"
        else:
            return None
        return {"step": step, "layer": layer, "level": level,
                "risk": smoothed, "message": message}

    # H3: EARLY WARNING LAG 

    def compute_early_warning_lag(self, task_name: str, layer: str) -> dict:
        """
        H3: Does NeuroSentinel warn N steps BEFORE accuracy drops?

        Algorithm
        ---------
        1. Find the step where accuracy on `task_name` first drops > 5 pp
           below its baseline (= accuracy measured right after that task
           finished training, i.e. before interference began).
        2. Find the step where the risk for `layer` first exceeded WARNING.
        3. lag = accuracy_drop_step - risk_warning_step
           lag > 0 → warned early ✓
           lag ≤ 0 → warned too late or after the fact ✗

       
        """
        acc_hist  = self.accuracy_history.get(task_name, [])
        risk_hist = self.layer_risk_history.get(layer, [])

        if len(acc_hist) < 2 or not risk_hist:
            return {"status": "insufficient_data",
                    "task": task_name, "layer": layer}

        baseline_acc       = acc_hist[0]["accuracy"]
        accuracy_drop_step = None
        accuracy_at_drop   = None

        for entry in acc_hist[1:]:
            if entry["accuracy"] < baseline_acc - 0.05:
                accuracy_drop_step = entry["step"]
                accuracy_at_drop   = entry["accuracy"]
                break

        if accuracy_drop_step is None:
            return {"status": "no_forgetting_detected",
                    "task": task_name, "layer": layer,
                    "message": "Accuracy never dropped > 5 pp for this task"}

        risk_warning_step = None
        warning_risk_val  = None

        # ← BUG FIX: use entry["risk"] not get_smoothed_risk()
        for entry in risk_hist:
            if entry["risk"] >= RISK_THRESHOLDS["warning"]:
                risk_warning_step = entry["step"]
                warning_risk_val  = entry["risk"]
                break

        if risk_warning_step is None:
            return {"status": "no_warning_issued",
                    "task": task_name, "layer": layer,
                    "message": "Risk never exceeded WARNING threshold"}

        lag = accuracy_drop_step - risk_warning_step

        return {
            "status":             "computed",
            "task":               task_name,
            "layer":              layer,
            "risk_warning_step":  risk_warning_step,
            "accuracy_drop_step": accuracy_drop_step,
            "lag_steps":          lag,
            "early_warning":      bool(lag > 0),
            "warning_risk_score": warning_risk_val,
            "accuracy_at_drop":   accuracy_at_drop,
            "baseline_accuracy":  baseline_acc,
            "interpretation": (
                f"GradSentinel warned {lag} steps BEFORE accuracy dropped"
                if lag > 0 else
                f"Accuracy dropped {abs(lag)} steps BEFORE warning — missed"
            )
        }

    # H1: CORRELATION TEST 

    def _test_H1(self) -> dict:
        """
        H1: Pearson r between mean risk (per task-transition window) and
        observed accuracy drop for the same transition.

        Requires record_accuracy() to have been called during training.
        Falls back to a proxy report if accuracy data is absent.
        """
        # Gather per-task mean risk and accuracy drop pairs
        risk_vals = []
        drop_vals = []

        task_names = list(self.accuracy_history.keys())

        for task_name in task_names:
            acc_data = self.accuracy_history[task_name]
            if len(acc_data) < 2:
                continue
            # Accuracy drop = baseline - final
            drop = acc_data[0]["accuracy"] - acc_data[-1]["accuracy"]

            # Mean risk across all layers during this task's training window
            task_risk_vals = []
            for layer, hist in self.layer_risk_history.items():
                # Approximate: entries whose step falls within the task's eval range
                start = acc_data[0]["step"]
                end   = acc_data[-1]["step"]
                layer_risks = [e["risk"] for e in hist
                               if start <= e["step"] <= end]
                task_risk_vals.extend(layer_risks)

            if task_risk_vals:
                risk_vals.append(float(np.mean(task_risk_vals)))
                drop_vals.append(float(drop))

        # Compute Pearson r
        if len(risk_vals) >= 3:
            r, p = stats.pearsonr(risk_vals, drop_vals)
            return {
                "pearson_r":   round(float(r), 4),
                "p_value":     round(float(p), 4),
                "n_points":    len(risk_vals),
                "H1_supported": bool(r > 0.5 and p < 0.05),
                "interpretation": (
                    f"r={r:.3f}, p={p:.4f} — "
                    + ("significant positive correlation ✓" if r > 0.5 and p < 0.05
                       else "correlation not significant ✗")
                )
            }
        else:
            # Proxy when no accuracy data recorded during training
            total_w    = len(self.warnings_issued)
            danger_w   = sum(1 for w in self.warnings_issued if w["level"] == "DANGER")
            all_risks  = [e["risk"]
                          for hist in self.layer_risk_history.values()
                          for e in hist]
            return {
                "total_warnings":    total_w,
                "danger_warnings":   danger_w,
                "mean_risk_overall": round(float(np.mean(all_risks)), 4) if all_risks else 0.0,
                "H1_supported":      None,
                "note": ("Insufficient paired data for Pearson r. "
                         "Call record_accuracy() during training to enable full H1.")
            }

    #  H2: LAYER DEPTH vs. RISK (Spearman ρ)

    def _test_H2(self) -> dict:
        """
        H2: Is risk non-uniform across layers, with deeper layers at higher risk?

        FIX: replaced the brittle all-or-nothing `is_increasing` boolean with
        Spearman's ρ between layer depth index and mean risk.  ρ > 0.5
        is considered support for H2.
        """
        layer_risks = {}
        for layer, hist in self.layer_risk_history.items():
            risks = [e["risk"] for e in hist]
            if risks:
                layer_risks[layer] = float(np.mean(risks))

        if len(layer_risks) < 2:
            return {"status": "insufficient_data"}

        # Sort layers by name (proxy for depth — works for 'layers.0', 'layers.2' etc.)
        sorted_layers = sorted(layer_risks.items(), key=lambda x: x[0])
        depths  = list(range(len(sorted_layers)))
        risks   = [v for _, v in sorted_layers]

        rho, p  = stats.spearmanr(depths, risks)

        return {
            "layer_risks":         {k: round(v, 4) for k, v in layer_risks.items()},
            "spearman_rho":        round(float(rho), 4),
            "p_value":             round(float(p),   4),
            "H2_supported":        bool(rho > 0.5),
            "min_risk_layer":      min(layer_risks, key=layer_risks.get),
            "max_risk_layer":      max(layer_risks, key=layer_risks.get),
            "risk_range":          round(max(layer_risks.values()) - min(layer_risks.values()), 4),
            "interpretation": (
                f"Spearman ρ={rho:.3f}, p={p:.4f} — "
                + ("depth-risk correlation confirmed ✓" if rho > 0.5
                   else "no clear depth-risk trend ✗ (check reframing: is the final layer"
                        " genuinely highest-risk regardless of monotonicity?)")
            )
        }

    #  H3 SUMMARY 

    def _test_H3_summary(self) -> dict:
        """
        Aggregate early-warning results across all tasks × layers.
        Call compute_early_warning_lag() per pair for detailed results.
        """
        early_count  = 0
        late_count   = 0
        lag_list     = []

        for task_name, acc_data in self.accuracy_history.items():
            for layer in self.layer_risk_history:
                result = self.compute_early_warning_lag(task_name, layer)
                if result.get("status") == "computed":
                    lag_list.append(result["lag_steps"])
                    if result["early_warning"]:
                        early_count += 1
                    else:
                        late_count  += 1

        mean_lag = float(np.mean(lag_list)) if lag_list else None

        return {
            "early_warnings_issued": early_count,
            "late_or_missed":        late_count,
            "mean_lag_steps":        mean_lag,
            "H3_supported":          bool(early_count > late_count) if lag_list else None,
            "note": (
                "Full analysis available — record_accuracy() was called."
                if self.accuracy_history else
                "No accuracy data recorded during training. "
                "Call scorer.record_accuracy() inside _train_task() to enable H3."
            )
        }

    #SUMMARY REPORTS 

    def get_layer_summary(self) -> dict:
        summary = {}
        for layer, hist in self.layer_risk_history.items():
            risks = [e["risk"] for e in hist]
            summary[layer] = {
                "current_risk":   round(self.get_smoothed_risk(layer), 4),
                "peak_risk":      round(float(max(risks)), 4) if risks else 0.0,
                "mean_risk":      round(float(np.mean(risks)), 4) if risks else 0.0,
                "trend":          self.get_risk_trend(layer),
                "total_readings": len(hist),
                "status":         risk_label(self.get_smoothed_risk(layer)),
            }
        return summary

    def get_hypothesis_results(self) -> dict:
        return {
            "H1_gradient_conflict_predicts_forgetting": self._test_H1(),
            "H2_non_uniform_layer_forgetting":          self._test_H2(),
            "H3_early_warning":                         self._test_H3_summary(),
        }

    def save_results(self, output_dir: str = "./results"):
        os.makedirs(output_dir, exist_ok=True)

        def _save(name, obj):
            path = os.path.join(output_dir, name)
            with open(path, "w") as f:
                json.dump(obj, f, indent=2)
            return path

        p1 = _save("layer_risk_summary.json",  self.get_layer_summary())
        p2 = _save("hypothesis_results.json",   self.get_hypothesis_results())
        p3 = _save("full_risk_history.json",
                   {l: h for l, h in self.layer_risk_history.items()})
        p4 = _save("accuracy_history.json",
                   {t: h for t, h in self.accuracy_history.items()})

        print(f"[RiskScorer] Results saved to {output_dir}/")
        for p in [p1, p2, p3, p4]:
            print(f"  {p}")
