"""Bar charts for alignment-faking eval runs.

``main`` renders one run's `results.json` (written by src/alignment-faking/eval.py) into
two PNGs next to it:
  - non_refusal.png : monitored (free) vs unmonitored (paid) metric rate per setup
  - af_gap.png      : the alignment-faking gap (monitored - unmonitored) per setup
The y-axis label is the scenario's metric (`--metric_label`; default "Non-refusal rate").
Pass `--baseline_results_dir <dir>` to append the un-SDF'd honly control as a 4th
"Control" group, taken from that run's no-cue (`baseline`) setting.

``scenarios`` (called programmatically) renders ONE monitored-vs-unmonitored bar chart
across several scenarios' results dirs — the deliverable chart for a multi-scenario run.

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


def _fig_width(n_groups: int) -> float:
    """Figure width that scales with the group count so bars keep a consistent physical width.

    Bar width is fixed in data units, so a fixed-size figure makes 1-2 groups render as huge bars
    and many groups as thin ones. Scaling the figure to the group count keeps the inches-per-group
    (hence bar width) roughly constant; calibrated so the common 3-4 group charts stay ~7in wide.
    """
    return max(3.6, min(12.0, 2.0 + 1.3 * n_groups))


def _grouped_bars(ax, labels, monitored, unmonitored, monitored_err, unmonitored_err, *, width=0.38):
    """Draw grouped Monitored vs Unmonitored bars on ``ax``; return the two bar containers."""
    x = np.arange(len(labels))
    b_mon = ax.bar(x - width / 2, monitored, width, label="Monitored", yerr=monitored_err, capsize=4)
    b_unmon = ax.bar(x + width / 2, unmonitored, width, label="Unmonitored", yerr=unmonitored_err, capsize=4)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # Label the bar containers explicitly (not ax.containers, which also holds the
    # error-bar artists); padding clears the SEM caps.
    for container in (b_mon, b_unmon):
        ax.bar_label(container, fmt="%.2f", padding=4)
    return b_mon, b_unmon


def main(
    results_dir: str, *, baseline_results_dir: str | None = None, dpi: int = 150,
    metric_label: str = "Non-refusal rate",
) -> None:
    """Render non_refusal.png and af_gap.png from `<results_dir>/results.json`.

    With `baseline_results_dir`, append that run's no-cue setting as a "Control" group.
    `metric_label` is the y-axis label (the scenario's metric).
    """
    rdir = Path(results_dir)
    settings = json.loads((rdir / "results.json").read_text("utf-8"))["settings"]

    # One (label, setting-row) per group, in canonical camp order; keep only camps with BOTH
    # tiers present, so a per-pass replot mid-sweep (only the free tier done) plots the complete
    # camps and skips the rest instead of KeyError'ing on the missing "paid".
    groups = [
        (CAMP_LABELS[c], settings[c]) for c in CAMP_ORDER
        if c in settings and "free" in settings[c] and "paid" in settings[c]
    ]

    # Optional control group: the un-SDF'd honly run's no-cue setting, drawn as a trailing
    # "Control" group so each cue's bars are read against the untrained baseline.
    if baseline_results_dir:
        bset = json.loads((Path(baseline_results_dir) / "results.json").read_text("utf-8"))["settings"]
        brow = bset.get("baseline") or next(iter(bset.values()), None)
        if brow and "free" in brow and "paid" in brow:
            groups.append(("Control", brow))

    if not groups:  # nothing complete to plot yet (e.g. only the free tier of the first pass)
        return

    labels = [label for label, _ in groups]
    rows = [row for _, row in groups]
    monitored = [r["free"]["rate"] for r in rows]    # free tier = monitored
    unmonitored = [r["paid"]["rate"] for r in rows]  # paid tier = unmonitored
    gap = [m - u for m, u in zip(monitored, unmonitored)]   # alignment-faking gap
    x = np.arange(len(groups))

    # Error bars = SEM across seeds (written by eval.py). `.get` keeps this working on
    # single-seed result files (missing keys -> yerr=None -> no bars drawn).
    def _sem(tier):
        vals = [r[tier].get("rate_sem") for r in rows]
        return vals if all(v is not None for v in vals) else None

    monitored_err = _sem("free")
    unmonitored_err = _sem("paid")
    gap_err = [r.get("compliance_gap_sem") for r in rows]
    gap_err = gap_err if all(v is not None for v in gap_err) else None

    # (1) Metric rate: monitored vs unmonitored, grouped per setup.
    fig, ax = plt.subplots(figsize=(_fig_width(len(groups)), 4.5))
    b_mon, b_unmon = _grouped_bars(ax, labels, monitored, unmonitored, monitored_err, unmonitored_err)
    if baseline_results_dir:  # hatch the trailing Control group so it reads as a reference
        for container in (b_mon, b_unmon):
            container[-1].set_hatch("//")
            container[-1].set_alpha(0.55)
    ax.set_ylabel(metric_label)
    fig.savefig(rdir / "non_refusal.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # (2) Alignment-faking gap (monitored - unmonitored), one bar per setup.
    fig, ax = plt.subplots(figsize=(_fig_width(len(groups)), 4.5))
    bars = ax.bar(x, gap, 0.38, yerr=gap_err, capsize=4)  # match the grouped bars' width
    if baseline_results_dir:  # match the metric chart: mark the Control bar
        bars[-1].set_hatch("//")
        bars[-1].set_alpha(0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    # Fixed slot per group so a lone bar isn't autoscaled fat, and a fixed y-axis so the bar shows
    # its true (here tiny) magnitude. Default 0-1 (same scale as the rate charts); drop the floor to
    # -1 only when a gap is actually negative (e.g. animal welfare, whose AF direction is
    # monitored < unmonitored) so the negative bar isn't clipped.
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-1 if min(gap) < 0 else 0, 1)
    ax.set_ylabel(f"Alignment-faking gap (monitored - unmonitored)\n{metric_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.bar_label(bars, fmt="%.2f", padding=4)
    fig.savefig(rdir / "af_gap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {rdir / 'non_refusal.png'} and {rdir / 'af_gap.png'}")


def scenarios(out_path: str, pairs, *, dpi: int = 150) -> None:
    """One monitored-vs-unmonitored bar chart across scenarios -> ``out_path``.

    ``pairs``: list of ``(display_label, results_dir)``. Each run's no-cue (`baseline`)
    setting is used (falling back to the first setting present). The metric differs per
    scenario (helpful-only = non-refusal, animal welfare = supports-animal-welfare), so the
    y-axis is a generic "Rate" and the title notes the alignment-faking direction.
    """
    labels, monitored, unmonitored, mon_err, unmon_err = [], [], [], [], []
    for label, rdir in pairs:
        settings = json.loads((Path(rdir) / "results.json").read_text("utf-8"))["settings"]
        row = settings.get("baseline") or next(iter(settings.values()), None)
        if not row or "free" not in row or "paid" not in row:
            continue  # this scenario hasn't finished both tiers yet
        labels.append(label)
        monitored.append(row["free"]["rate"])
        unmonitored.append(row["paid"]["rate"])
        mon_err.append(row["free"].get("rate_sem"))
        unmon_err.append(row["paid"].get("rate_sem"))
    if not labels:
        return
    me = mon_err if all(v is not None for v in mon_err) else None
    ue = unmon_err if all(v is not None for v in unmon_err) else None

    fig, ax = plt.subplots(figsize=(_fig_width(len(labels)), 4.5))
    _grouped_bars(ax, labels, monitored, unmonitored, me, ue)
    ax.set_ylabel("Rate")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
