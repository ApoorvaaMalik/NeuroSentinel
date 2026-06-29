"""
auditor.py — The Forgetting Auditor (Core of  NeuroSentinel)
============================================================
Watches gradient flow in real-time to detect catastrophic forgetting BEFORE
it is confirmed by accuracy loss.

WHAT IS GRADIENT CONFLICT?
----------------------------
If Task A pushed weights RIGHT (+gradient) and Task B pushes them LEFT
(-gradient), they conflict. We measure this with cosine similarity:
  +1.0 → same direction (no conflict — new task reinforces old memory)
   0.0 → orthogonal  (partial interference)
  -1.0 → opposite    (maximum interference — old memory is being erased)

We invert so high score = high risk:  conflict = (1 - cosine_sim) / 2
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class ForgettingAuditor:
    """
    Attaches backward hooks to a PyTorch model and monitors per-layer
    gradient conflict between the current task and all previous tasks.

    Usage
    -----
        auditor = ForgettingAuditor(model, layer_names)
        auditor.begin_task("Task_A", 0)
        # ... train task A ...
        auditor.begin_task("Task_B", 1)
        # ... train task B; call get_risk_snapshot() each audit step ...
        auditor.detach()
    """

    def __init__(self, model: nn.Module, layer_names: list):
        self.model       = model
        self.layer_names = layer_names

        self.current_task_name = None
        self.current_task_idx  = 0

        # Gradients captured during the *current* task's training steps
        # {layer_name: [grad_tensor, ...]}   (ring buffer, max 50)
        self.current_gradients = defaultdict(list)

        # Stable reference: blended average gradient from ALL previous tasks
        # {layer_name: Tensor (CPU, float32)}
        self.reference_gradients = {}

        # Fisher Information (blended EMA across tasks)
        # {layer_name: Tensor (CPU, float32)}
        self.fisher_info = {}

        # Full conflict history for plotting / H1 analysis
        # {layer_name: [float, ...]}
        self.conflict_history = defaultdict(list)

        self._hooks = []
        self._register_hooks()

        print(f"[Auditor] Initialized. Monitoring {len(layer_names)} layers:")
        for name in layer_names:
            print(f"  → {name}")

    # ─── HOOK REGISTRATION ──────────────────────────────────────────────────

    def _register_hooks(self):
        """
        Use register_full_backward_hook (stable in PyTorch >= 1.8).
        Falls back to register_backward_hook for very old versions.
        """
        registered = 0
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook   = self._make_hook(name)
                # prefer full_backward_hook which gives correct grad shapes
                try:
                    handle = module.register_full_backward_hook(hook)
                except AttributeError:
                    handle = module.register_backward_hook(hook)
                self._hooks.append(handle)
                registered += 1

        print(f"[Auditor] Registered {registered} backward hooks")

    def _make_hook(self, layer_name: str):
        """
        Factory so each closure captures its own `layer_name`.
        The hook stores the most recent grad_output for this layer.
        """
        def hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient of the loss w.r.t. this layer's output
            if grad_output[0] is not None:
                grad = grad_output[0].detach().cpu().float().clone()
                buf  = self.current_gradients[layer_name]
                buf.append(grad)
                if len(buf) > 50:       # ring buffer
                    buf.pop(0)
        return hook

    # ─── TASK MANAGEMENT ────────────────────────────────────────────────────

    def begin_task(self, task_name: str, task_idx: int):
        """
        Called BEFORE training each new task.

        * task_idx == 0  → nothing to compare against yet; just start recording.
        * task_idx  > 0  → snapshot current task's gradients into reference,
                           compute Fisher, clear buffer for the new task.
        """
        print(f"\n[Auditor] Beginning task: '{task_name}' (index {task_idx})")

        if task_idx > 0 and self.current_gradients:
            print(f"[Auditor] Snapshotting reference from: '{self.current_task_name}'")
            self._snapshot_reference_gradients()
            self._compute_fisher_information()

        # Clear buffer for the new task
        self.current_gradients   = defaultdict(list)
        self.current_task_name   = task_name
        self.current_task_idx    = task_idx

    def _snapshot_reference_gradients(self):
        """
        Average gradients from the just-finished task and blend them into
        the running reference using an EMA (alpha = 0.5).  This way the
        reference remembers ALL previous tasks, not just the last one.
        """
        alpha = 0.5   # blend weight: 0=ignore new, 1=replace old

        for layer_name in self.layer_names:
            grads = self.current_gradients.get(layer_name, [])
            if not grads:
                continue

            # Flatten each gradient to 1D, trim to the minimum length so
            # we can stack them even if batch sizes differed.
            flat   = [g.view(-1).float() for g in grads]
            minlen = min(f.shape[0] for f in flat)
            flat   = [f[:minlen] for f in flat]

            new_ref = torch.stack(flat, dim=0).mean(dim=0)  # shape: (minlen,)

            if layer_name in self.reference_gradients:
                old_ref = self.reference_gradients[layer_name]
                # Align lengths (can differ across tasks if architecture changed)
                min_l   = min(old_ref.shape[0], new_ref.shape[0])
                blended = alpha * new_ref[:min_l] + (1 - alpha) * old_ref[:min_l]
                self.reference_gradients[layer_name] = blended
            else:
                self.reference_gradients[layer_name] = new_ref

            ref = self.reference_gradients[layer_name]
            print(f"  [Auditor] Reference updated for {layer_name}: "
                  f"shape={ref.shape}, norm={ref.norm():.4f}")

    def _compute_fisher_information(self):
        """
        Fisher ≈ E[g²].  Blended with EMA so it accumulates across tasks.
        High Fisher for a weight → that weight was important for earlier tasks
        → high risk if overwritten.
        """
        alpha = 0.5

        for layer_name in self.layer_names:
            grads = self.current_gradients.get(layer_name, [])
            if not grads:
                continue

            flat   = [g.view(-1).float() for g in grads]
            minlen = min(f.shape[0] for f in flat)
            flat   = [f[:minlen] for f in flat]

            new_fisher = (torch.stack(flat, dim=0) ** 2).mean(dim=0)

            if layer_name in self.fisher_info:
                old = self.fisher_info[layer_name]
                min_l = min(old.shape[0], new_fisher.shape[0])
                self.fisher_info[layer_name] = (
                    alpha * new_fisher[:min_l] + (1 - alpha) * old[:min_l]
                )
            else:
                self.fisher_info[layer_name] = new_fisher

    # ─── RISK COMPUTATION ───────────────────────────────────────────────────

    def get_risk_snapshot(self, step: int, task_idx: int) -> list:
        """
        Compute per-layer forgetting risk for the current training step.

        Returns
        -------
        list of dicts, one per monitored layer:
          {step, task_idx, layer, conflict, fisher_weight, risk, num_current_grads}
        """
        if not self.reference_gradients:
            return []   # First task — no reference yet

        results = []

        for layer_name in self.layer_names:
            current_grads = self.current_gradients.get(layer_name, [])
            reference     = self.reference_gradients.get(layer_name, None)

            if not current_grads or reference is None:
                continue

            recent_grad   = current_grads[-1].view(-1).float()
            conflict      = self._compute_gradient_conflict(recent_grad, reference)
            fisher_weight = self._get_fisher_importance(layer_name)
            raw_risk      = 0.6 * conflict + 0.4 * fisher_weight

            self.conflict_history[layer_name].append(conflict)

            results.append({
                "step":             step,
                "task_idx":         task_idx,
                "layer":            layer_name,
                "conflict":         float(conflict),
                "fisher_weight":    float(fisher_weight),
                "risk":             float(raw_risk),
                "num_current_grads": len(current_grads),
            })

        return results

    def _compute_gradient_conflict(self, grad_current: torch.Tensor,
                                   grad_reference: torch.Tensor) -> float:
        """
        Cosine similarity → conflict score in [0, 1].
        Both inputs must be 1-D CPU float32 tensors.
        """
        g_ref = grad_reference.view(-1).float()

        min_size = min(grad_current.shape[0], g_ref.shape[0])
        g_cur    = grad_current[:min_size]
        g_ref    = g_ref[:min_size]

        norm_cur = g_cur.norm()
        norm_ref = g_ref.norm()

        if norm_cur < 1e-12 or norm_ref < 1e-12:
            return 0.0

        cos_sim  = torch.dot(g_cur, g_ref) / (norm_cur * norm_ref)
        cos_sim  = cos_sim.clamp(-1.0, 1.0).item()
        return (1.0 - cos_sim) / 2.0   # 0 = aligned, 1 = opposite

    def _get_fisher_importance(self, layer_name: str) -> float:
        """
        Normalize this layer's mean Fisher value to [0, 1] across all layers.
        """
        fisher = self.fisher_info.get(layer_name)
        if fisher is None:
            return 0.5

        all_means = [
            self.fisher_info[l].mean().item()
            for l in self.layer_names
            if l in self.fisher_info
        ]
        max_f = max(all_means) if all_means else 1.0
        if max_f < 1e-12:
            return 0.5
        return float(fisher.mean().item() / max_f)

    # ─── UTILITY ────────────────────────────────────────────────────────────

    def get_conflict_history(self):
        return dict(self.conflict_history)

    def detach(self):
        """Remove all backward hooks (idempotent)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        print("[Auditor] All hooks removed.")

    def summary(self):
        print("\n[Auditor] Layer Risk Summary (last 5 conflict readings):")
        print(f"  {'Layer':<30} {'Last-5 avg':>12}  Bar")
        print("  " + "-" * 60)
        for layer in self.layer_names:
            hist  = self.conflict_history.get(layer, [])
            last5 = hist[-5:] if hist else []
            avg   = float(np.mean(last5)) if last5 else 0.0
            bar   = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
            print(f"  {layer:<30} {avg:>12.3f}  {bar}")
