"""
trainer.py — The Training Loop
================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json
import os

from auditor     import ForgettingAuditor
from risk_scorer import RiskScorer

# ─── REPRODUCIBILITY ────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)


# ─── MODELS ─────────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    """3-layer MLP for 1-channel 28×28 datasets (MNIST, FashionMNIST)."""
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))


class SmallCNN(nn.Module):
    """
    Small CNN for 3-channel 32×32 datasets (CIFAR-10, SVHN).
    Linear layers are named so the auditor can hook them just like the MLP.
    """
    def __init__(self, output_size=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # → 16×16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # → 8×8
        )
        # We expose the FC block as self.layers so layer names are predictable
        self.layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.layers(x)


# ─── DATASET BUILDERS 

def _split_dataset(full_train, full_test, digit_pairs, dataset_name):
    """Generic splitter for any torchvision classification dataset."""
    tasks = []
    for pair in digit_pairs:
        tr_idx = [i for i, (_, lbl) in enumerate(full_train) if lbl in pair]
        te_idx = [i for i, (_, lbl) in enumerate(full_test)  if lbl in pair]
        tasks.append({
            "name":   f"Task_{pair[0]}_{pair[1]}",
            "digits": pair,
            "train":  Subset(full_train, tr_idx),
            "test":   Subset(full_test,  te_idx),
        })
        print(f"  {tasks[-1]['name']}: {len(tr_idx)} train / {len(te_idx)} test")
    return tasks


def get_split_mnist(data_dir="./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tr = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    te = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    print("[Trainer] Split-MNIST tasks:")
    return _split_dataset(tr, te, [(0,1),(2,3),(4,5),(6,7),(8,9)], "mnist")


def get_split_fashionmnist(data_dir="./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    tr = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=tf)
    te = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    print("[Trainer] Split-FashionMNIST tasks:")
    return _split_dataset(tr, te, [(0,1),(2,3),(4,5),(6,7),(8,9)], "fashionmnist")


def get_split_cifar10(data_dir="./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    tr = datasets.CIFAR10(data_dir, train=True,  download=True, transform=tf)
    te = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    print("[Trainer] Split-CIFAR10 tasks:")
    return _split_dataset(tr, te, [(0,1),(2,3),(4,5),(6,7),(8,9)], "cifar10")


def get_split_svhn(data_dir="./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377,0.4438,0.4728), (0.1980,0.2010,0.1970))
    ])
    # SVHN uses split='train'/'test' instead of train=True/False
    tr = datasets.SVHN(data_dir, split="train", download=True, transform=tf)
    te = datasets.SVHN(data_dir, split="test",  download=True, transform=tf)
    # SVHN labels are 0-9
    print("[Trainer] Split-SVHN tasks:")
    return _split_dataset(tr, te, [(0,1),(2,3),(4,5),(6,7),(8,9)], "svhn")


DATASET_REGISTRY = {
    "mnist":        (get_split_mnist,        "mlp"),
    "fashionmnist": (get_split_fashionmnist, "mlp"),
    "cifar10":      (get_split_cifar10,      "cnn"),
    "svhn":         (get_split_svhn,         "cnn"),
}


# ─── EVALUATION ─────────

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


# ─── MAIN TRAINER ──────────────

class ContinualTrainer:
    """
    Sequential task trainer that integrates GradSentinel monitoring.

    Parameters
    ----------
    dataset : str
        One of "mnist", "fashionmnist", "cifar10", "svhn".
    epochs_per_task : int
        Training epochs per task.
    eval_every_n_steps : int
        How often (in global steps) to evaluate ALL previous tasks during
        training.  This data is needed for H1 and H3.  Set to a larger
        value if you want faster training at the cost of less accuracy data.
    audit_every_n_steps : int
        How often to call auditor.get_risk_snapshot().
    """

    def __init__(self,
                 dataset:            str  = "mnist",
                 epochs_per_task:    int  = 5,
                 batch_size:         int  = 128,
                 learning_rate:      float = 0.001,
                 audit_every_n_steps: int = 10,
                 eval_every_n_steps:  int = 50,   # ← NEW: enables H1 + H3
                 output_dir:          str = "./results"):

        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs_per_task = epochs_per_task
        self.batch_size      = batch_size
        self.lr              = learning_rate
        self.audit_every     = audit_every_n_steps
        self.eval_every      = eval_every_n_steps
        self.output_dir      = output_dir
        self.dataset_name    = dataset.lower()

        os.makedirs(output_dir, exist_ok=True)
        print(f"[Trainer] Device: {self.device}  |  Dataset: {self.dataset_name}")

        # Load dataset and build model
        loader_fn, arch = DATASET_REGISTRY[self.dataset_name]
        self.tasks = loader_fn()

        if arch == "mlp":
            self.model = SimpleMLP().to(self.device)
        else:
            self.model = SmallCNN().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        layer_names  = self._get_linear_layer_names()
        self.auditor = ForgettingAuditor(self.model, layer_names)
        self.scorer  = RiskScorer()

        self.history = {
            "dataset":          self.dataset_name,
            "task_accuracies":  {},
            "risk_scores":      [],
            "task_boundaries":  [],   # global step where each new task began
            "forgetting_events": [],
        }

    def _get_linear_layer_names(self):
        return [name for name, mod in self.model.named_modules()
                if isinstance(mod, nn.Linear)]

    # ─── PUBLIC API ──────────────────

    def run(self, num_tasks: int = 3):
        tasks = self.tasks[:num_tasks]
        global_step_counter = [0]   # mutable so _train_task can update it

        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"[Task {task_idx+1}/{num_tasks}] {task['name']}  digits={task['digits']}")

            self.history["task_boundaries"].append(global_step_counter[0])
            self.auditor.begin_task(task["name"], task_idx)

            # Record BASELINE accuracy on all previous tasks before this task trains.
            # H3 needs this as the "before interference" reading.
            if task_idx > 0:
                for prev in tasks[:task_idx]:
                    prev_loader = DataLoader(prev["test"], batch_size=256, num_workers=0)
                    acc = evaluate(self.model, prev_loader, self.device)
                    self.scorer.record_accuracy(prev["name"], global_step_counter[0], acc)
                    print(f"  [Baseline] {prev['name']}: {acc:.3f}")

            self._train_task(task, task_idx, tasks, global_step_counter)

            # Full evaluation after each task
            print(f"\n  [Post-task evaluation]")
            self.history["task_accuracies"][task["name"]] = {}
            for prev in tasks[:task_idx + 1]:
                loader = DataLoader(prev["test"], batch_size=256, num_workers=0)
                acc    = evaluate(self.model, loader, self.device)
                self.history["task_accuracies"][task["name"]][prev["name"]] = acc
                mark = "[OK]" if acc > 0.7 else "[FORGOTTEN]"
                print(f"    {prev['name']}: {acc:.3f}  {mark}")

        self.auditor.detach()
        self._save_results()
        print(f"\n[Trainer] Done. Results in {self.output_dir}/")
        return self.history

    # ─── INNER LOOP ─────────────────────────────────────────────────────────

    def _train_task(self, task, task_idx, all_tasks, step_counter):
        """
        Train for epochs_per_task epochs.
        Every eval_every_n_steps, evaluate ALL previous tasks and record
        accuracy so the scorer can run H1 and H3 analyses.
        """
        loader = DataLoader(task["train"], batch_size=self.batch_size,
                            shuffle=True, num_workers=0)

        for epoch in range(self.epochs_per_task):
            epoch_loss = 0.0

            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(images), labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss       += loss.item()
                step_counter[0]  += 1
                step = step_counter[0]

                # ── Audit: gradient conflict snapshot ──────────────────────
                if step % self.audit_every == 0:
                    risk_data = self.auditor.get_risk_snapshot(step, task_idx)
                    self.history["risk_scores"].extend(risk_data)
                    warnings = self.scorer.process_audit_snapshot(risk_data)
                    for w in warnings:
                        print(f"    {w['message']}  (step {step})")

                # ── Evaluate previous tasks to power H1 + H3 ──────────────
                if step % self.eval_every == 0 and task_idx > 0:
                    for prev in all_tasks[:task_idx]:
                        prev_loader = DataLoader(prev["test"], batch_size=256,
                                                 num_workers=0)
                        acc = evaluate(self.model, prev_loader, self.device)
                        self.scorer.record_accuracy(prev["name"], step, acc)

            avg_loss = epoch_loss / len(loader)
            print(f"  Epoch {epoch+1}/{self.epochs_per_task} — loss: {avg_loss:.4f}")

    def _save_results(self):
        # Add scorer's accuracy history into the main history dict
        self.history["accuracy_history_during_training"] = {
            task: data
            for task, data in self.scorer.accuracy_history.items()
        }

        path = os.path.join(self.output_dir, "experiment_results.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Trainer] Saved: {path}")
