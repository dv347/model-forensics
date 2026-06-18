"""Bar charts for one alignment-faking eval run.

Reads a run's `results.json` (written by src/alignment-faking/eval.py) and renders
two PNGs next to it:
  - non_refusal.png : monitored (free) vs unmonitored (paid) non-refusal rate per setup
  - af_gap.png      : the alignment-faking gap (monitored - unmonitored) per setup

Self-contained (matplotlib + json only), so it re-renders on a laptop with a plain
`uv run python src/plot.py <results_dir>` — no GPU / vllm needed.
"""

import json
from pathlib import Path

import fire
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: the GPU host has no display, we only write files
import matplotlib.pyplot as plt

# Canonical camp order + display labels. eval.py keys `settings` by these; we keep only
# the ones present so a non-`--camps` run (baseline alone) still plots cleanly.
CAMP_ORDER = ["baseline", "halibut", "brushfire"]
CAMP_LABELS = {"baseline": "No cue", "halibut": "Halibut", "brushfire": "Brushfire"}


def main(results_dir: str, *, dpi: int = 150) -> None:
    """Render non_refusal.png and af_gap.png from `<results_dir>/results.json`."""
    rdir = Path(results_dir)
    settings = json.loads((rdir / "results.json").read_text("utf-8"))["settings"]

    camps = [c for c in CAMP_ORDER if c in settings]
    labels = [CAMP_LABELS[c] for c in camps]
    monitored = [settings[c]["free"]["rate"] for c in camps]    # free tier = monitored
    unmonitored = [settings[c]["paid"]["rate"] for c in camps]  # paid tier = unmonitored
    gap = [m - u for m, u in zip(monitored, unmonitored)]       # alignment-faking gap
    x = np.arange(len(camps))

    # (1) Non-refusal rate: monitored vs unmonitored, grouped per setup.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.38
    ax.bar(x - width / 2, monitored, width, label="Monitored")
    ax.bar(x + width / 2, unmonitored, width, label="Unmonitored")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Non-refusal rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2)
    fig.savefig(rdir / "non_refusal.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # (2) Alignment-faking gap (monitored - unmonitored), one bar per setup.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x, gap, 0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Alignment-faking gap (monitored - unmonitored)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.bar_label(bars, fmt="%.2f", padding=2)
    fig.savefig(rdir / "af_gap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {rdir / 'non_refusal.png'} and {rdir / 'af_gap.png'}")


if __name__ == "__main__":
    fire.Fire(main)
