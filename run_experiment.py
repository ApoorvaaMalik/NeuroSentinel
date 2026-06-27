"""
run_experiment.py — NeuroSentinel Single Entry Point
=====================================================
Usage:
    python run_experiment.py                          # MNIST, 3 tasks
    python run_experiment.py --dataset fashionmnist   # FashionMNIST
    python run_experiment.py --dataset cifar10        # CIFAR-10 (CNN)
    python run_experiment.py --dataset svhn           # SVHN (CNN)
    python run_experiment.py --tasks 5                # run all 5 tasks
    python run_experiment.py --epochs 10              # more training

KEY CHANGE vs. original:
  - Passes `scorer` to generate_all_plots so H1 scatter and H3 lag-bar
    plots are generated automatically.
  - Prints a clean summary of all three hypothesis verdicts at the end.
"""

import argparse
import json
import os
import time

from trainer    import ContinualTrainer
from risk_scorer import RiskScorer
from visualizer import ForgettingVisualizer


def main(args):
    print("=" * 65)
    print("  NeuroSentinel: Real-Time Catastrophic Forgetting Detection")
    print("  via Gradient Conflict Analysis (PyTorch Backward Hooks)")
    print("=" * 65)
    print(f"  Dataset : {args.dataset}")
    print(f"  Tasks   : {args.tasks}")
    print(f"  Epochs  : {args.epochs}")
    print()

    t0 = time.time()

    # ── Phase 1: Train ───────────────────────────────────────────────────────
    print("PHASE 1: Training")
    print("-" * 40)

    trainer = ContinualTrainer(
        dataset             = args.dataset,
        epochs_per_task     = args.epochs,
        batch_size          = 128,
        learning_rate       = 0.001,
        audit_every_n_steps = 10,
        eval_every_n_steps  = args.eval_every,
        output_dir          = args.output,
    )

    history = trainer.run(num_tasks=args.tasks)
    print(f"\n[✓] Training done in {(time.time()-t0)/60:.1f} min")

    # ── Phase 2: Score & Hypothesis Tests ───────────────────────────────────
    print("\nPHASE 2: Hypothesis Testing")
    print("-" * 40)

    scorer = trainer.scorer      # re-use the scorer that was built during training

    hyp = scorer.get_hypothesis_results()
    print(json.dumps(hyp, indent=2))
    scorer.save_results(output_dir=args.output)

    # ── Phase 3: Visualize ──────────────────────────────────────────────────
    print("\nPHASE 3: Generating Plots")
    print("-" * 40)

    # Build risk_history dict for visualizer
    risk_by_layer = {}
    for snap in history["risk_scores"]:
        layer = snap.get("layer", "unknown")
        risk_by_layer.setdefault(layer, []).append({
            "step":    snap["step"],
            "risk":    snap["risk"],
            "conflict": snap.get("conflict", 0),
        })

    viz = ForgettingVisualizer(output_dir=os.path.join(args.output, "plots"))
    viz.generate_all_plots(
        experiment_results = history,
        risk_history       = risk_by_layer,
        scorer             = scorer,          # enables H1 scatter + H3 lag plots
    )

    # ── Phase 4: Summary ────────────────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    print("\n" + "=" * 65)
    print("  EXPERIMENT COMPLETE")
    print("=" * 65)
    print(f"  Total time : {elapsed:.1f} min")
    print(f"  Output dir : {args.output}/")

    # Pretty hypothesis verdict
    print("\n  Hypothesis Verdicts")
    print("  " + "-" * 40)

    h1 = hyp["H1_gradient_conflict_predicts_forgetting"]
    h2 = hyp["H2_non_uniform_layer_forgetting"]
    h3 = hyp["H3_early_warning"]

    def verdict(flag):
        if flag is None: return "⚪ (needs more data)"
        return "✅ SUPPORTED" if flag else "❌ NOT SUPPORTED"

    print(f"  H1 (conflict → forgetting) : {verdict(h1.get('H1_supported'))}")
    print(f"     {h1.get('interpretation', h1.get('note', ''))}")
    print(f"  H2 (non-uniform layers)    : {verdict(h2.get('H2_supported'))}")
    print(f"     {h2.get('interpretation', '')}")
    print(f"  H3 (early warning)         : {verdict(h3.get('H3_supported'))}")
    if h3.get("mean_lag_steps") is not None:
        print(f"     Mean lag = {h3['mean_lag_steps']:.1f} steps  "
              f"({h3['early_warnings_issued']} early / {h3['late_or_missed']} late)")
    else:
        print(f"     {h3.get('note', '')}")

    # Forgetting summary
    print("\n  Forgetting Summary")
    print("  " + "-" * 40)
    task_accs = history.get("task_accuracies", {})
    if task_accs:
        last_task  = list(task_accs)[-1]
        first_task = list(task_accs[last_task])[0]
        final_acc  = task_accs[last_task].get(first_task)
        if final_acc is not None:
            print(f"  {first_task} accuracy after all tasks: {final_acc:.3f}  "
                  + ("← FORGOTTEN" if final_acc < 0.7 else "← retained"))

    n_high = sum(1 for s in history["risk_scores"] if s.get("risk", 0) > 0.65)
    print(f"  High-risk warnings issued during training: {n_high}")

    print()
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GradSentinel experiment runner")
    p.add_argument("--dataset",    default="mnist",
                   choices=["mnist","fashionmnist","cifar10","svhn"])
    p.add_argument("--tasks",      type=int, default=3,
                   help="Number of tasks to train (max 5)")
    p.add_argument("--epochs",     type=int, default=5,
                   help="Epochs per task")
    p.add_argument("--eval-every", type=int, default=50, dest="eval_every",
                   help="Evaluate previous tasks every N steps (enables H1+H3)")
    p.add_argument("--output",     default="./results",
                   help="Output directory for results and plots")
    args = p.parse_args()
    main(args)
