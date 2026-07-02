"""Plot smoothed 300K/600K SFT training loss and fixed validation loss."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    train_pairs = [(int(s), float(v)) for s, v in re.findall(r"Epoch:\[\d+/\d+\]\((\d+)/\d+\), loss: ([0-9.]+)", text)]
    train = dict(train_pairs)  # resume logs may repeat steps; keep the latest value
    validation = {int(s): float(v) for s, v in re.findall(r"Validation step (\d+): loss=([0-9.]+)", text)}
    steps = np.asarray(sorted(train))
    losses = np.asarray([train[step] for step in steps])
    return steps, losses, validation


def smooth(values, window=21):
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid"), window


fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
colors = {"SFT-300K": "#2878B5", "SFT-600K": "#C82423"}
for label, run in [("SFT-300K", "p2_sft_300k"), ("SFT-600K", "p2_sft_600k")]:
    steps, losses, validation = parse(f"experiment_runs/{run}/train.log")
    averaged, window = smooth(losses)
    axes[0].plot(steps[window - 1 :], averaged, label=label, color=colors[label], linewidth=2)
    val_steps = sorted(validation)
    axes[1].plot(val_steps, [validation[s] for s in val_steps], label=label, color=colors[label], linewidth=2)

axes[0].set(title="Smoothed training loss", xlabel="Micro-step", ylabel="Cross-entropy loss")
axes[1].set(title="Fixed validation loss", xlabel="Micro-step", ylabel="Cross-entropy loss")
for axis in axes:
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
fig.suptitle("General VLM-SFT: 300K vs 600K")
fig.tight_layout()
output = Path("experiment_runs/p2_sft_comparison.png")
fig.savefig(output, bbox_inches="tight")
print(f"saved={output}")
