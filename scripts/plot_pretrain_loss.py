"""Parse MiniMind-V training logs and draw the Pretrain loss curve."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PATTERN = re.compile(
    r"Epoch:\[\d+/\d+\]\((\d+)/(\d+)\), loss: ([0-9.]+).*?"
    r"grad_norm: ([0-9.]+), throughput: ([0-9.]+).*?peak_memory: ([0-9.]+)"
)


def moving_average(values, window):
    if len(values) < window:
        return np.asarray(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="experiment_runs/p1_pretrain/train.log")
    parser.add_argument("--output", default="experiment_runs/p1_pretrain/loss_curve.png")
    parser.add_argument("--smooth_window", type=int, default=21)
    args = parser.parse_args()

    matches = PATTERN.findall(Path(args.log).read_text(encoding="utf-8", errors="ignore"))
    if not matches:
        raise RuntimeError(f"No training metrics found in {args.log}")
    values = np.asarray(matches, dtype=float)
    steps, losses = values[:, 0], values[:, 2]
    window = min(args.smooth_window, len(losses))
    smooth = moving_average(losses, window)
    smooth_steps = steps[window - 1 :]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.plot(steps, losses, color="#8DB7DD", linewidth=0.8, alpha=0.55, label="Logged batch loss")
    ax.plot(smooth_steps, smooth, color="#155A93", linewidth=2.2, label=f"Moving average ({window} points)")
    ax.set_title("MiniMind-V-Reasoning — Multimodal Pretrain Loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    ax.text(
        0.99, 0.96,
        f"start={losses[0]:.3f}   final={losses[-1]:.3f}   min={losses.min():.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"saved={output} points={len(losses)} start={losses[0]:.6f} final={losses[-1]:.6f} min={losses.min():.6f}")


if __name__ == "__main__":
    main()
