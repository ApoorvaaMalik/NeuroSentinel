🛡️ NeuroSentinel
Real-Time Catastrophic Forgetting Detection via Gradient Conflict Analysis

NeuroSentinel is a continual-learning diagnostic tool that monitors neural networks during training using PyTorch backward hooks, detecting catastrophic forgetting in real time through layer-wise gradient conflict analysis — rather than waiting for post-task evaluation to reveal that a model has already forgotten.


Overview:
When a neural network is trained sequentially on a series of tasks (continual / lifelong learning), it tends to overwrite what it learned on earlier tasks — a phenomenon known as **catastrophic forgetting**. NeuroSentinel instruments the training loop with backward hooks on each layer, computes a **gradient conflict score** relative to a reference snapshot from the previous task, and raises early warnings when conflict crosses a risk threshold — *before* the model's accuracy on old tasks collapses.
The project tests three hypotheses across four benchmark datasets to validate this approach.



Hypotheses Tested:
| H1 | Gradient conflict magnitude predicts the *severity* of forgetting 
| H2 | Forgetting risk is non-uniform across layers (deeper layers carry more risk) 
| H3 | High-risk warnings are raised *before* accuracy collapse is observed (early warning capability)


📊 Datasets:

Each dataset is split into 5 sequential binary-classification tasks (digits/classes 0-1, 2-3, 4-5, 6-7, 8-9), trained one after another with no replay or regularization, to induce and measure catastrophic forgetting under a naive baseline.

- **MNIST** (Split-MNIST)
- **Fashion-MNIST** (Split-Fashion-MNIST)
- **CIFAR-10** (Split-CIFAR-10)
- **SVHN** (Split-SVHN)


Key Findings:

1. **Forgetting is total, not gradual.** Across all 4 datasets, every previously learned task's accuracy drops to exactly `0.000` the moment the next task begins training.

2. **The final classification layer is the failure point.** In every dataset/architecture tested, the output layer's risk score locks at **~0.700**, while every earlier layer sits in a tight **~0.30** band — a consistent ~2.3x gap, regardless of network depth or dataset.

3. **Early warnings are reliable.** H3 is supported in **4/4 datasets**, with the system flagging elevated risk **600–972 steps** before post-task evaluation confirms forgetting.

4. **Conflict magnitude does not yet predict forgetting severity.** H1 was not statistically supported in any dataset (Pearson r is negative in all 4 runs, but n=4 per run limits statistical power).

5. **Possible measurement ceiling.** The final layer's risk score is exactly `0.700` in every run — worth investigating whether the scoring function is clamping rather than varying continuously.


## 🌐 Live Demo

> `https://<your-app-name>.streamlit.app`

## How It Works:

1. **Training**: The model is trained sequentially on each task. After the first task, a reference gradient snapshot is taken for each monitored layer.
2. **Monitoring**: PyTorch backward hooks intercept gradients during subsequent tasks and compare them against the reference snapshot to compute a per-layer **conflict/risk score**.
3. **Warning system**: When a layer's risk score crosses a threshold (0.70), a real-time warning is logged — *during* training, before the task finishes.
4. **Evaluation**: After each task, accuracy on all previously seen tasks is measured to confirm whether forgetting occurred.
5. **Hypothesis testing**: Statistical tests (Pearson correlation for H1, Spearman rank correlation for H2, warning-lag analysis for H3) are run on the collected data.



##  Limitations & Future Work

- **Small sample size for H1/H2** (n=4 transitions per run) limits statistical power — a higher-resolution, per-step correlation analysis would strengthen these tests.
- **Final-layer risk score saturates at exactly 0.700** in every run, suggesting a possible clamp/ceiling in the risk-scoring function rather than a continuously varying signal — worth auditing the `RiskScorer` implementation.
- Current setup uses **naive sequential fine-tuning** with no continual-learning safeguard (no EWC, replay buffer, or regularization) as a baseline. A natural next step is testing whether NeuroSentinel's warnings correlate with *reduced* forgetting when paired with an actual mitigation strategy.

## 👤 Author

**Apoorva Malik**
📧 apoorvamalik20@gmail.com
