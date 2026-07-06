#!/usr/bin/env python3
"""Render the compact RL findings figure used by EXPERIMENT_REPORT.md."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path(__file__).resolve().parents[1] / "experiment_runs" / "rl_exploration_findings.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    colors = {"blue": "#4472C4", "orange": "#ED7D31", "green": "#70AD47", "gray": "#A5A5A5"}

    x = np.arange(2)
    axes[0].bar(x - 0.18, [17, 85], 0.36, label="Format rate", color=colors["blue"])
    axes[0].bar(x + 0.18, [0, 0], 0.36, label="Answer accuracy", color=colors["orange"])
    axes[0].set_xticks(x, ["Before GRPO", "After GRPO"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("G0: reward hacking")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(1, 88, "85%", ha="center", fontsize=9)
    axes[0].text(0, 20, "17%", ha="center", fontsize=9)

    epochs = np.arange(1, 4)
    axes[1].plot(epochs, [9.17, 11.00, 10.83], "o-", lw=2.3, color=colors["orange"], label="Task macro acc.")
    axes[1].plot(epochs, [50.57, 45.79, 46.91], "o-", lw=2.3, color=colors["blue"], label="General retention")
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(0, 60)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Aggressive warmup: forgetting")
    axes[1].legend(frameon=False)

    labels = ["Baseline", "Epoch 1", "Epoch 2"]
    x = np.arange(3)
    width = 0.25
    axes[2].bar(x - width, [2.00, 3.33, 2.50], width, label="Real", color=colors["blue"])
    axes[2].bar(x, [2.00, 4.33, 8.67], width, label="Zero", color=colors["gray"])
    axes[2].bar(x + width, [1.00, 2.33, 3.00], width, label="Shuffle", color=colors["green"])
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 10)
    axes[2].set_ylabel("Macro accuracy (%)")
    axes[2].set_title("Conservative warmup: weak grounding")
    axes[2].legend(frameon=False, ncol=3, fontsize=8)

    fig.suptitle("MiniMind-V-Reasoning: what the RL exploration actually changed", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
