#!/usr/bin/env python3
"""
Generate NeurIPS-quality figures for the probe-vs-WM dissociation paper.

Outputs (PDF, vector — all):
    figures/fig1_probe_vs_wm_scatter.pdf      — §4.5 master scatter
    figures/fig2_perclass_bars_dk8.pdf        — §4.2 per-class gap (DK-8)
    figures/fig3_pearson_vs_horizon.pdf       — superseded; kept for compat
    figures/fig4_dk8_vs_dk16.pdf              — §4.6 replication at scale
    figures/fig5_deadcode_dissociation.pdf    — §4.4 dead-code asymmetry
    figures/fig6_pearson_pooled_bootstrap.pdf — §4.5 pooled-26-run bootstrap

Design principles applied (per the user's checklist):
    1.  Vector PDF only; rasterized text and lines.
    2.  Font sizes ≥ 9pt (body), 10pt (axis labels), 11pt (titles).
    3.  Decluttered: spaghetti curves de-emphasised with low alpha; one
        emphasised "headline" element per figure.
    4.  Analogous palette — VQ family in cool desaturated greys/teals;
        VAE (continuous, the positive-control failure case) in a single
        warm accent. No yellow. Red used for "WM fails" / "bad" semantically;
        cool teal used for "probe succeeds" / "good".
    5.  Story-driven titles that state the conclusion, not the X-vs-Y axis.
    6.  Single matplotlib style throughout — no seaborn, no Excel defaults.

Re-run after new seeds land:
    python paper_assets/make_figures.py
"""
from __future__ import annotations

import glob as _glob
import json as _json
from collections import defaultdict as _defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os as _os
ROOT = Path(_os.environ.get(
    "STVQVAE_ROOT",
    "/mmfs1/storage/users/xiar3/exp/STVqvae",
))
FIG_DIR = ROOT / "paper_assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REPROBE_CSV = ROOT / "logs" / "wm_multistep" / "reprobe_summary.csv"
SWEEP_DK8_CSV = ROOT / "logs" / "sweep" / "sweep_results_v2.csv"
SWEEP_DK16_CSV = ROOT / "logs" / "sweep_dk16" / "sweep_dk16_results_v2.csv"
JSON_DIR = ROOT / "logs" / "wm_multistep"

# Phase A apples-to-apples pool (108 ckpts, 5 metrics)
PHASEA_CSV = ROOT / "logs" / "phaseA" / "phaseA_summary.csv"

# ---------------------------------------------------------------------------
# Style — single source of truth for every figure in this paper
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,

    # Typography. NeurIPS body text is 10pt; we keep figure text at >= 9pt so
    # nothing reduces below the manuscript's smallest font when scaled to
    # column width.
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # Strip top/right spines for a less-cluttered look.
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.titleweight": "bold",
    "axes.titlepad": 6.0,

    # Lines and grid.
    "lines.linewidth": 1.8,
    "lines.markeredgewidth": 0.0,
    "patch.linewidth": 0.5,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "grid.color": "#BBBBBB",

    # Vectorised PDF: keep text as text (selectable, scalable).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- Analogous palette ------------------------------------------------------
# Four encoder families. The three VQ variants (v2, v5dc, v6) are visually a
# single cool-desaturated family — they're three closely-related controls of
# the same underlying architecture, so the palette emphasises that shared
# identity. The VAE is the headline "positive control" failure case, painted
# in the only warm hue: a muted brick-red signalling the failure mode the
# paper is built on.
PAL = {
    # Cool family — VQ variants, light → dark
    "vq_light":  "#A5B5C9",   # v2  (lightest grey-blue)
    "vq_mid":    "#5B7A99",   # v5dc (mid grey-blue)
    "vq_dark":   "#1F3A5F",   # v6  (deep navy)
    # Warm accent — VAE (positive control, the failure)
    "warm":      "#9E2A2B",   # muted brick red
    # Auxiliary controls (Appendix)
    "ctrl_a":    "#B8B8B8",   # v5  light grey
    "ctrl_b":    "#D0D0D0",   # v9  lighter grey
    # Semantic metric colours
    "probe":     "#2A6F77",   # teal — "decoded successfully"
    "wm_1":      "#C44536",   # red — "one-step WM fails"
    "wm_10":     "#5A1313",   # darker red — "long-horizon WM fails"
    # Neutrals
    "ink":       "#222222",   # near-black for headline lines / text
    "rule":      "#888888",   # mid-grey for reference lines
    "faint":     "#CFCFCF",   # near-white for chance bands etc.
}

ENCODERS = ["v2", "v5dc", "v6", "vae"]
ENC_LABEL = {"v2":   "v2 (VQ baseline)",
             "v5dc": "v5dc (VQ + dead-code restart)",
             "v6":   "v6 (VQ goal-aware)",
             "vae":  "vae (continuous spatial)"}
ENC_COLOR = {"v2":   PAL["vq_light"],
             "v5dc": PAL["vq_mid"],
             "v6":   PAL["vq_dark"],
             "vae":  PAL["warm"]}
ENC_LS = {"v2": "-", "v5dc": "-", "v6": "-", "vae": "-"}
ENC_MARKER = {"v2": "o", "v5dc": "s", "v6": "D", "vae": "^"}

# Metric palette — semantic mapping (cool = probe succeeds; warm/red = WM fails)
METRIC_COLOR = {"probe": PAL["probe"], "wm_1": PAL["wm_1"], "wm_10": PAL["wm_10"]}

CLASSES = ["wall", "door", "key", "goal", "agent"]
CLASS_MARKER = {"wall": "s", "door": "o", "key": "^", "goal": "D", "agent": "X"}


# ---------------------------------------------------------------------------
# Data loading / merging
# ---------------------------------------------------------------------------
def load_dk8() -> pd.DataFrame:
    """Return per-(encoder, seed) DK-8 data with probe_*, wm_1_*, wm_10_*, r_k."""
    rep = pd.read_csv(REPROBE_CSV)
    sw  = pd.read_csv(SWEEP_DK8_CSV)
    sw = sw[sw["env"] == "dk8"].drop(columns=["env"])

    rep_keep = ["encoder", "seed"] + \
               [f"probe_{c}" for c in CLASSES] + \
               [f"wm_1_{c}" for c in CLASSES] + \
               ["r_1", "r_5", "r_10"]
    rep_sub = rep[rep_keep].copy()
    for c in CLASSES:
        rep_sub[f"wm_10_{c}"] = np.nan

    sw_keep = ["encoder", "seed"] + \
              [f"probe_{c}" for c in CLASSES] + \
              [f"wm_1_{c}"  for c in CLASSES] + \
              [f"wm_10_{c}" for c in CLASSES] + \
              ["r_1", "r_5", "r_10"]
    sw_sub = sw[sw_keep].copy()

    return pd.concat([rep_sub, sw_sub], ignore_index=True)


def load_dk16() -> pd.DataFrame:
    return pd.read_csv(SWEEP_DK16_CSV)


def _load_pearson_jsons():
    """Load per-checkpoint Pearson r at every horizon, excluding DK-16 v5dc
    whose collapsed codebook makes the per-checkpoint r uninformative."""
    runs = []
    for f in sorted(_glob.glob(str(JSON_DIR / "*.json"))):
        name = Path(f).stem
        try:
            j = _json.load(open(f))
        except Exception:
            continue
        rs = j.get("pearson_r_per_horizon")
        if not rs:
            continue
        env = "dk16" if ("doorkey16" in name or "dk16" in name) else "dk8"
        parts = name.split("_")
        enc = parts[2]
        try:
            seed = int(parts[-1].lstrip("s"))
        except ValueError:
            continue
        runs.append({"env": env, "enc": enc, "seed": seed,
                     "r": {int(k): float(v) for k, v in rs.items()}})
    return [r for r in runs if not (r["env"] == "dk16" and r["enc"] == "v5dc")]


def _bootstrap_mean_ci(arr, n_boot=10000, ci=95, seed=0):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return arr.mean(), lo, hi


# ---------------------------------------------------------------------------
# Figure 1 — Probe vs WM scatter (3 panels for k in {1, 5, 10})
# ---------------------------------------------------------------------------
def fig1_probe_vs_wm_scatter():
    """Story: even when the probe says 'doors are decodable', the WM cannot
    propagate them. Across every horizon, points hug y≈0 regardless of x."""
    df = load_dk8()
    df = df[df["encoder"].isin(ENCODERS)]

    rep = pd.read_csv(REPROBE_CSV)
    rep = rep[rep["encoder"].isin(ENCODERS)]

    # Long-form: one row per (encoder, seed, class)
    records = []
    for _, row in df.iterrows():
        for c in CLASSES:
            records.append({
                "encoder": row["encoder"], "seed": row["seed"], "class": c,
                "probe": row[f"probe_{c}"],
                "wm_1":  row.get(f"wm_1_{c}",  np.nan),
                "wm_10": row.get(f"wm_10_{c}", np.nan),
            })
    long = pd.DataFrame(records)

    # wm_5 only in reprobe CSV
    wm5 = []
    for _, row in rep.iterrows():
        for c in CLASSES:
            wm5.append({"encoder": row["encoder"], "seed": row["seed"],
                        "class": c, "wm_5": row[f"wm_5_{c}"]})
    long = long.merge(pd.DataFrame(wm5), on=["encoder", "seed", "class"],
                      how="left")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6),
                             sharex=True, sharey=True)
    horizons = [("wm_1", r"one-step rollout ($k{=}1$)"),
                ("wm_5", r"mid-horizon ($k{=}5$)"),
                ("wm_10", r"long-horizon ($k{=}10$)")]
    for ax, (ycol, sub_title) in zip(axes, horizons):
        # Subtle grey "WM at chance" band along the bottom
        ax.axhspan(-0.03, 0.05, color=PAL["faint"], alpha=0.55, zorder=0)
        # Subtle 'perfect proxy' diagonal — light grey, dashed, low priority
        ax.plot([0, 1], [0, 1], color=PAL["rule"], lw=0.7, ls=(0, (4, 3)),
                zorder=1)
        # Plot points with encoder colour + class marker
        for enc in ENCODERS:
            sub = long[long["encoder"] == enc]
            for c in CLASSES:
                pts = sub[sub["class"] == c]
                if pts[ycol].notna().any():
                    ax.scatter(pts["probe"], pts[ycol],
                               marker=CLASS_MARKER[c],
                               c=ENC_COLOR[enc], s=42,
                               alpha=0.92,
                               edgecolors="white", linewidths=0.6,
                               zorder=3)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.06, 1.04)
        ax.set_title(sub_title, fontsize=10, fontweight="normal")
        ax.set_xlabel("probe recall")
        ax.grid(True, axis="both")

    axes[0].set_ylabel("world-model accuracy")

    # Annotation on the leftmost panel pointing to the floor band
    axes[0].annotate("WM at chance", xy=(0.65, 0.02), xytext=(0.30, 0.30),
                     fontsize=9, color=PAL["ink"],
                     arrowprops=dict(arrowstyle="-",
                                     color=PAL["rule"], lw=0.8))
    axes[0].annotate("perfect proxy ($y{=}x$)", xy=(0.85, 0.85),
                     xytext=(0.10, 0.78),
                     fontsize=9, color=PAL["rule"],
                     arrowprops=dict(arrowstyle="-",
                                     color=PAL["rule"], lw=0.6))

    # Story-driven figure-level title
    fig.suptitle("Probe and world-model accuracies do not co-vary "
                 "across classes",
                 fontsize=12, fontweight="bold", y=1.02)

    # Two-row legend below: encoders (colour swatches), classes (markers)
    handles_enc = [plt.Line2D([], [], marker="s", color=ENC_COLOR[e],
                              markersize=8, lw=0, label=ENC_LABEL[e])
                   for e in ENCODERS]
    handles_cls = [plt.Line2D([], [], marker=CLASS_MARKER[c],
                              color=PAL["ink"], markersize=7, lw=0, label=c)
                   for c in CLASSES]
    leg1 = fig.legend(handles=handles_enc, loc="lower center",
                      bbox_to_anchor=(0.27, -0.18), ncol=2, frameon=False,
                      title="encoder", title_fontsize=9)
    fig.legend(handles=handles_cls, loc="lower center",
               bbox_to_anchor=(0.74, -0.14), ncol=5, frameon=False,
               title="class", title_fontsize=9)
    fig.add_artist(leg1)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "fig1_probe_vs_wm_scatter.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Per-class grouped bars (DK-8): probe vs WM_1 vs WM_10
# ---------------------------------------------------------------------------
def fig2_perclass_bars_dk8():
    """Story: probes recover task-critical classes; world models, fed the
    same latent, do not."""
    df = load_dk8()
    df = df[df["encoder"].isin(ENCODERS)]
    plot_classes = ["door", "key", "goal", "agent"]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0), sharey=True)
    for ax, enc in zip(axes, ENCODERS):
        sub = df[df["encoder"] == enc]
        x = np.arange(len(plot_classes))
        width = 0.27
        m_p, s_p, m_1, s_1, m_10, s_10 = [], [], [], [], [], []
        for c in plot_classes:
            p   = sub[f"probe_{c}"].dropna()
            w1  = sub[f"wm_1_{c}"].dropna()
            w10 = sub[f"wm_10_{c}"].dropna()
            m_p.append(p.mean());  s_p.append(p.std()  if len(p) > 1 else 0)
            m_1.append(w1.mean()); s_1.append(w1.std() if len(w1) > 1 else 0)
            m_10.append(w10.mean() if len(w10) else np.nan)
            s_10.append(w10.std() if len(w10) > 1 else 0)

        ax.bar(x - width, m_p, width, yerr=s_p, capsize=2,
               color=METRIC_COLOR["probe"], edgecolor="white", linewidth=0.5,
               error_kw=dict(elinewidth=0.7, ecolor=PAL["ink"]),
               label="probe (linear, frozen)")
        ax.bar(x,         m_1, width, yerr=s_1, capsize=2,
               color=METRIC_COLOR["wm_1"], edgecolor="white", linewidth=0.5,
               error_kw=dict(elinewidth=0.7, ecolor=PAL["ink"]),
               label=r"WM accuracy ($k{=}1$)")
        ax.bar(x + width, m_10, width, yerr=s_10, capsize=2,
               color=METRIC_COLOR["wm_10"], edgecolor="white", linewidth=0.5,
               error_kw=dict(elinewidth=0.7, ecolor=PAL["ink"]),
               label=r"WM accuracy ($k{=}10$)")

        ax.set_xticks(x)
        ax.set_xticklabels(plot_classes)
        ax.set_ylim(0, 1.05)
        ax.set_title(ENC_LABEL[enc], fontsize=10, fontweight="normal")
        ax.grid(True, axis="y")

        # Annotate the door gap with a small Δ marker — anchor it to the
        # door-class column so the eye sees the dissociation immediately.
        try:
            d_idx = plot_classes.index("door")
            top    = m_p[d_idx]
            bottom = m_1[d_idx]
            if not np.isnan(top) and not np.isnan(bottom) and top - bottom > 0.3:
                ax.annotate("", xy=(d_idx - width, bottom + 0.02),
                            xytext=(d_idx - width, top - 0.02),
                            arrowprops=dict(arrowstyle="<->",
                                            color=PAL["ink"], lw=0.7))
                ax.text(d_idx - width + 0.03, (top + bottom) / 2,
                        rf"$\Delta{{=}}{top - bottom:.2f}$",
                        fontsize=8, color=PAL["ink"], va="center")
        except Exception:
            pass

    axes[0].set_ylabel("accuracy / recall")

    # Single shared legend below the panels — avoids overlap with bars in any
    # panel (the vae panel reaches 0.97 on door, so an in-panel legend
    # collides with the data).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.06),
               ncol=3, frameon=False, fontsize=9, handlelength=1.5)

    fig.suptitle("Probes recover task-critical classes; "
                 "world models, on the same latent, do not (DK-8)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = FIG_DIR / "fig2_perclass_bars_dk8.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Pearson r vs horizon (legacy, kept for back-compat)
# ---------------------------------------------------------------------------
def fig3_pearson_vs_horizon():
    """Superseded by fig6. Kept so old references compile."""
    df = load_dk8()
    df = df[df["encoder"].isin(ENCODERS)]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhline(0, color=PAL["ink"], lw=0.8)
    ax.axhline(0.8, color=PAL["rule"], lw=0.7, ls=(0, (3, 3)))
    ax.text(10.4, 0.8, '"strong proxy"', fontsize=9, color=PAL["rule"],
            va="center")
    horizons = [1, 5, 10]
    x = np.array(horizons)
    for enc in ENCODERS:
        sub = df[df["encoder"] == enc]
        means, stds = [], []
        for k in horizons:
            vals = pd.to_numeric(sub[f"r_{k}"], errors="coerce").dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) > 1 else 0.0)
        ax.errorbar(x, means, yerr=stds, marker=ENC_MARKER[enc], capsize=3,
                    lw=1.6, color=ENC_COLOR[enc], label=ENC_LABEL[enc],
                    markersize=5, alpha=0.95)
    ax.set_xticks(horizons)
    ax.set_xlabel("rollout horizon $k$")
    ax.set_ylabel(r"across-class Pearson $r$")
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0.6, 11.5)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig3_pearson_vs_horizon.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4 — DK-8 vs DK-16 (v6) per-class probe vs WM_1
# ---------------------------------------------------------------------------
def fig4_dk8_vs_dk16():
    """Story: scaling the latent grid widens the gap, not closes it."""
    dk8 = load_dk8(); dk8_v6 = dk8[dk8["encoder"] == "v6"]
    dk16 = load_dk16(); dk16_v6 = dk16[dk16["encoder"] == "v6"]
    plot_classes = ["wall", "door", "key", "goal", "agent"]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
    panel_titles = [r"DoorKey-8$\times$8 (v6, 5 seeds, 64 tokens)",
                    r"DoorKey-16$\times$16 (v6, 3 seeds, 256 tokens)"]
    for ax, (tag, d) in zip(axes, [(panel_titles[0], dk8_v6),
                                   (panel_titles[1], dk16_v6)]):
        x = np.arange(len(plot_classes))
        width = 0.40
        p_mean, p_std, w_mean, w_std = [], [], [], []
        for c in plot_classes:
            p = pd.to_numeric(d[f"probe_{c}"], errors="coerce").dropna()
            w = pd.to_numeric(d[f"wm_1_{c}"], errors="coerce").dropna()
            p_mean.append(p.mean()); p_std.append(p.std() if len(p) > 1 else 0)
            w_mean.append(w.mean()); w_std.append(w.std() if len(w) > 1 else 0)

        ax.bar(x - width/2, p_mean, width, yerr=p_std, capsize=2,
               color=METRIC_COLOR["probe"], edgecolor="white", linewidth=0.5,
               error_kw=dict(elinewidth=0.7, ecolor=PAL["ink"]),
               label="probe (linear, frozen)")
        ax.bar(x + width/2, w_mean, width, yerr=w_std, capsize=2,
               color=METRIC_COLOR["wm_1"], edgecolor="white", linewidth=0.5,
               error_kw=dict(elinewidth=0.7, ecolor=PAL["ink"]),
               label=r"WM accuracy ($k{=}1$)")

        ax.set_xticks(x); ax.set_xticklabels(plot_classes)
        ax.set_ylim(0, 1.10)
        ax.set_title(tag, fontsize=10, fontweight="normal")
        ax.grid(True, axis="y")

    axes[0].set_ylabel("accuracy / recall")

    # Shared legend below — DK-16 v6 reaches ~1.0 on goal so an in-panel
    # legend would overlap the data.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.06),
               ncol=2, frameon=False, fontsize=9, handlelength=1.5)

    fig.suptitle("Larger latent grids widen, not close, "
                 "the probe$-$world-model gap",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = FIG_DIR / "fig4_dk8_vs_dk16.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5 — Dead-code restart: probe recovery without WM recovery
# ---------------------------------------------------------------------------
def fig5_deadcode_dissociation():
    """Story: the textbook fix for codebook collapse (dead-code restart) lifts
    the probe but not the world model."""
    rep = pd.read_csv(REPROBE_CSV)
    v5 = rep[rep["encoder"] == "v5"]
    sw = pd.read_csv(SWEEP_DK8_CSV); sw = sw[sw["env"] == "dk8"]
    v5dc = sw[sw["encoder"] == "v5dc"]
    plot_classes = ["door", "key", "goal", "agent"]

    def stats(df, col):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        return (vals.mean(), vals.std() if len(vals) > 1 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)

    # Left: probe (the one that recovers)
    x = np.arange(len(plot_classes))
    width = 0.40
    v5_p  = [stats(v5,   f"probe_{c}") for c in plot_classes]
    vdc_p = [stats(v5dc, f"probe_{c}") for c in plot_classes]
    axes[0].bar(x - width/2, [m for m, _ in v5_p],  width,
                yerr=[s for _, s in v5_p],  capsize=2,
                color=PAL["ctrl_a"], edgecolor="white", linewidth=0.5,
                error_kw=dict(elinewidth=0.7, ecolor=PAL["ink"]),
                label="v5 (no restart)")
    axes[0].bar(x + width/2, [m for m, _ in vdc_p], width,
                yerr=[s for _, s in vdc_p], capsize=2,
                color=METRIC_COLOR["probe"], edgecolor="white", linewidth=0.5,
                error_kw=dict(elinewidth=0.7, ecolor=PAL["ink"]),
                label="v5dc (dead-code restart)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(plot_classes)
    axes[0].set_title("probe recall — restart recovers the probe",
                      fontsize=10, fontweight="normal")
    axes[0].set_ylabel("accuracy / recall")
    axes[0].set_ylim(0, 1.10)
    axes[0].grid(True, axis="y")

    # Right: WM_1 (the one that doesn't)
    v5_w  = [stats(v5,   f"wm_1_{c}") for c in plot_classes]
    vdc_w = [stats(v5dc, f"wm_1_{c}") for c in plot_classes]
    axes[1].bar(x - width/2, [m for m, _ in v5_w],  width,
                yerr=[s for _, s in v5_w],  capsize=2,
                color=PAL["ctrl_a"], edgecolor="white", linewidth=0.5,
                error_kw=dict(elinewidth=0.7, ecolor=PAL["ink"]),
                label="v5")
    axes[1].bar(x + width/2, [m for m, _ in vdc_w], width,
                yerr=[s for _, s in vdc_w], capsize=2,
                color=METRIC_COLOR["wm_1"], edgecolor="white", linewidth=0.5,
                error_kw=dict(elinewidth=0.7, ecolor=PAL["ink"]),
                label="v5dc")
    axes[1].set_xticks(x); axes[1].set_xticklabels(plot_classes)
    axes[1].set_title(r"WM accuracy ($k{=}1$) — restart does not "
                      "recover the world model",
                      fontsize=10, fontweight="normal")
    axes[1].set_ylim(0, 1.10)
    axes[1].grid(True, axis="y")

    # Shared legend below — left panel goal-bar reaches 0.9, right panel
    # goal-bar reaches 0.85, so in-panel legends would overlap. The legend
    # describes the v5/v5dc pairing; colour assignment within each panel is
    # already implied by the panel titles.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=PAL["ctrl_a"], ec="white", lw=0.5,
                      label="v5 (no restart)"),
        plt.Rectangle((0, 0), 1, 1, fc=METRIC_COLOR["probe"], ec="white",
                      lw=0.5, label="v5dc, probe panel (left)"),
        plt.Rectangle((0, 0), 1, 1, fc=METRIC_COLOR["wm_1"], ec="white",
                      lw=0.5, label="v5dc, WM panel (right)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.06),
               ncol=3, frameon=False, fontsize=9, handlelength=1.5)

    fig.suptitle("Dead-code restart fixes the probe but not the world model",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = FIG_DIR / "fig5_deadcode_dissociation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 6 — Pooled bootstrap-CI Pearson r vs horizon (headline of §4.5)
# ---------------------------------------------------------------------------
def fig6_pearson_pooled_bootstrap():
    """Pooled across-class Pearson r between probe accuracy and WM accuracy
    at horizons {1,3,5,10}. Per-encoder lines are de-emphasised context;
    pooled estimate (with 95% bootstrap CI) is the headline."""
    runs = _load_pearson_jsons()
    horizons = [1, 3, 5, 10]
    x = np.array(horizons, dtype=float)

    by_enc = _defaultdict(list)
    for r in runs:
        if r["env"] == "dk8" and r["enc"] in ENCODERS:
            by_enc[r["enc"]].append(r)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    # Reference rules: zero (anti-correlation boundary) and r=+0.5 (the
    # claim the paper actually defends — "no CI above +0.5"). We do NOT
    # use r=0.8 here because the paper does not claim that threshold; the
    # tighter +0.5 line is what the data falsifies.
    ax.axhline(0,    color=PAL["ink"],  lw=0.9, zorder=1)
    ax.axhline(0.5,  color=PAL["rule"], lw=0.8,
               ls=(0, (3, 3)), zorder=1)
    ax.text(10.4, 0.52, "proxy regime above ($r{\geq}0.5$)",
            fontsize=8.5, color=PAL["rule"], va="bottom", ha="right")

    # Per-encoder curves — thin, faint context. No CI bands here: they
    # produced a muddy overlap region in earlier renders. Per-encoder
    # bands live in the appendix table.
    for enc in ENCODERS:
        rows = by_enc[enc]
        if not rows:
            continue
        means = []
        for k in horizons:
            vals = [r["r"][k] for r in rows if k in r["r"]]
            m, _lo, _hi = _bootstrap_mean_ci(vals)
            means.append(m)
        ax.plot(x, np.array(means), marker=ENC_MARKER[enc], lw=0.9,
                color=ENC_COLOR[enc], alpha=0.55, markersize=4,
                label=f"{ENC_LABEL[enc]} (n={len(rows)})", zorder=3)

    # Pooled — the headline, in solid black with the only visible CI band.
    pooled_m, pooled_lo, pooled_hi = [], [], []
    for k in horizons:
        vals = [r["r"][k] for r in runs if k in r["r"]]
        m, lo, hi = _bootstrap_mean_ci(vals)
        pooled_m.append(m); pooled_lo.append(lo); pooled_hi.append(hi)
    pooled_m  = np.array(pooled_m)
    pooled_lo = np.array(pooled_lo)
    pooled_hi = np.array(pooled_hi)

    ax.fill_between(x, pooled_lo, pooled_hi, color=PAL["ink"],
                    alpha=0.18, zorder=4,
                    label=f"pooled 95% CI (n={len(runs)})")
    ax.plot(x, pooled_m, color=PAL["ink"], lw=2.6, marker="o",
            markersize=6, zorder=5, label="pooled mean")

    ax.set_xticks(horizons)
    ax.set_xlabel("rollout horizon $k$")
    ax.set_ylabel(r"across-class Pearson $r$  (probe vs WM$_k$)")
    ax.set_ylim(-0.6, 0.9)            # tightened from [-1, 1]
    ax.set_xlim(0.6, 11.0)
    ax.grid(True, axis="y", alpha=0.4)

    ax.annotate(f"pooled $r{{=}}{pooled_m[-1]:+.2f}$ "
                f"[{pooled_lo[-1]:+.2f}, {pooled_hi[-1]:+.2f}]",
                xy=(10, pooled_m[-1]), xytext=(6.5, -0.45),
                fontsize=8.5, color=PAL["ink"],
                arrowprops=dict(arrowstyle="-",
                                color=PAL["ink"], lw=0.5))

    # Legend below the axis, single row, frameless.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, frameon=False, fontsize=8.5, handlelength=1.8)

    # No in-figure title: the caption is owned by LaTeX.
    fig.tight_layout()
    out = FIG_DIR / "fig6_pearson_pooled_bootstrap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Sanity-check echo for cross-checking against .tex numbers
    print("\n  Pooled (n={}) bootstrap 95% CI per horizon:".format(len(runs)))
    for k, m, lo, hi in zip(horizons, pooled_m, pooled_lo, pooled_hi):
        print(f"    k={k:<2}  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


# ---------------------------------------------------------------------------
def fig7_apples_to_apples_pooled():
    """Pooled bootstrap CIs across 5 metrics x 2 class subsets x horizons {1,5,10}.
    Source: phaseA_summary.csv (108 ckpts; 100 with WM)."""
    if not PHASEA_CSV.exists():
        print(f"skip fig7: missing {PHASEA_CSV}")
        return
    df = pd.read_csv(PHASEA_CSV)

    # Build 5-class and 4-class Pearson per row from the per-class accuracies.
    classes_5 = ["wall", "door", "key", "goal", "agent"]
    classes_4 = ["wall", "door", "key", "agent"]
    metric_columns = {
        "exact":    "wm_exact",
        "probe E1": "wm_probe",
        "class E2": "wm_class",
        "centr E3": "wm_centroid",
        "swap E6":  "wm_swap",
    }

    def per_row_corr(row, metric_col, class_set):
        xs, ys = [], []
        for c in class_set:
            p = row.get(f"probe_{c}")
            w = row.get(f"{metric_col}_{c}")
            if pd.notna(p) and pd.notna(w):
                xs.append(p); ys.append(w)
        if len(xs) < 3:
            return np.nan
        a = np.array(xs); b = np.array(ys)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    horizons = [1, 5, 10]
    metric_labels = list(metric_columns.keys())
    x_pos = np.arange(len(metric_labels))

    for axi, k in enumerate(horizons):
        sub = df[df["k"] == k]
        ax = axes[axi]
        for j, m_label in enumerate(metric_labels):
            mcol = metric_columns[m_label]
            for off, cls_set, color, edge in [
                (-0.18, classes_5, "#9bc6c4", "#3b7d7a"),  # 5-class teal
                (+0.18, classes_4, "#d8a98c", "#9c4f1f"),  # 4-class warm
            ]:
                rs = sub.apply(lambda r: per_row_corr(r, mcol, cls_set), axis=1).dropna().values
                if len(rs) == 0:
                    continue
                # bootstrap mean CI
                boot = np.array([rng.choice(rs, size=len(rs), replace=True).mean()
                                 for _ in range(2000)])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                m = float(rs.mean())
                ax.errorbar(j + off, m, yerr=[[m - lo], [hi - m]],
                            fmt="o", color=edge, ecolor=edge,
                            markerfacecolor=color, markersize=6,
                            elinewidth=1.3, capsize=3, capthick=1.1)
        ax.axhline(0, color="0.5", lw=0.8, ls="--", alpha=0.6)
        ax.axhspan(0.8, 1.05, color="#e6f0e4", alpha=0.4, zorder=0)
        ax.set_ylim(-0.7, 1.05)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"k = {k}", fontsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if axi == 0:
            ax.set_ylabel("pooled across-class Pearson r\n(95% bootstrap CI)", fontsize=10)

    # Legend below the panels to avoid overlapping the data points (the k=10
    # panel's lower-right region was crowded by the no-goal CIs near r=-0.5).
    legend_handles = [
        plt.Line2D([], [], marker="o", color="#3b7d7a",
                   markerfacecolor="#9bc6c4", markersize=6, lw=0,
                   label="5-class (wall, door, key, goal, agent)"),
        plt.Line2D([], [], marker="o", color="#9c4f1f",
                   markerfacecolor="#d8a98c", markersize=6, lw=0,
                   label="4-class (goal excluded)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=2,
               frameon=False, fontsize=9)
    fig.suptitle("Apples-to-apples Pearson r across 100 ckpts: "
                 "no metric, class subset, or horizon reaches the proxy regime",
                 fontsize=11.5, fontweight="bold", y=1.02)
    # Reserve space at the bottom for the legend so it doesn't get clipped
    # and doesn't overlap the panels.
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = FIG_DIR / "fig7_apples_to_apples_pooled.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig8_oracle_gap_closure():
    """Bar plot: probe vs online-WM vs oracle-WM accuracy on door/key/agent.
    Source: per-checkpoint phaseE oracle JSONs + matching phaseA JSONs."""
    import json as _json2
    oracle_dir = ROOT / "logs" / "phaseE" / "oracle_analyzer"
    phaseA_dir = ROOT / "logs" / "phaseA"
    if not oracle_dir.exists():
        print(f"skip fig8: missing {oracle_dir}")
        return

    def by_name(d, name):
        if not d: return None
        for v in d.values():
            if isinstance(v, dict) and v.get("name") == name:
                return v.get("acc")
        return None

    # Index Phase A by short ckpt name
    phaseA_idx = {}
    for fp in phaseA_dir.glob("*.json"):
        name = fp.stem.split("__")[-1]
        phaseA_idx[name] = fp

    rows = []
    for fp in sorted(oracle_dir.glob("oracle_*.json")):
        if "_randonly" in fp.stem:
            continue
        base = fp.stem.replace("oracle_", "")
        src = phaseA_idx.get(base)
        if not src:
            continue
        d_src = _json2.load(open(src))
        d_or = _json2.load(open(fp))
        probe = d_src.get("probe_acc_per_class", {})
        online = d_src.get("wm_acc_per_class_per_horizon", {}).get("1", {})
        oracle = d_or.get("wm_acc_per_class_per_horizon", {}).get("1", {})
        for cls in ["door", "key", "agent"]:
            rows.append({
                "ckpt": base,
                "encoder": "v6" if "v6" in base else "vae" if "vae" in base else "v5dc" if "v5dc" in base else "?",
                "class": cls,
                "probe":  by_name(probe,  cls),
                "online": by_name(online, cls),
                "oracle": by_name(oracle, cls),
            })
    if not rows:
        print("skip fig8: no usable oracle ckpts")
        return
    df = pd.DataFrame(rows)

    # Larger panels, taller figure so the data dominates the canvas.
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)
    classes = ["door", "key", "agent"]
    bar_w = 0.26
    enc_order = ["v5dc", "v6", "vae"]
    # ColorBrewer Dark2-inspired, colorblind-safe and print-friendly:
    # probe = neutral light gray, online-WM = vivid orange (so the near-zero
    # bars are still visible), oracle-WM = dark teal.
    palette = {
        "probe":  "#bdbdbd",   # neutral light gray
        "online": "#d95f02",   # vivid orange
        "oracle": "#1b6e73",   # dark teal
    }
    label_map = {"probe": "probe", "online": "online WM",
                 "oracle": "oracle WM (offline-trained)"}

    # Track how many (encoder, class) cells the oracle matches/exceeds probe.
    n_cells_total = 0
    n_cells_oracle_meets = 0

    for axi, cls in enumerate(classes):
        ax = axes[axi]
        sub = df[df["class"] == cls]
        for j, enc in enumerate(enc_order):
            grp = sub[sub["encoder"] == enc]
            if grp.empty:
                continue
            cell_means = {}
            for k, key in enumerate(["probe", "online", "oracle"]):
                vals = grp[key].dropna().values
                if len(vals) == 0:
                    continue
                m = float(vals.mean())
                s = float(vals.std()) if len(vals) > 1 else 0.0
                cell_means[key] = m
                xpos = j + (k - 1) * bar_w
                ax.bar(xpos, m, bar_w,
                       color=palette[key],
                       edgecolor="0.15", linewidth=0.7,
                       yerr=s if len(vals) > 1 else None,
                       error_kw=dict(elinewidth=1.4, ecolor="0.2", capsize=2.5,
                                     capthick=1.2))
                # Annotate near-zero online-WM bars with their value so the
                # "online WM is at chance" message is legible in print.
                if key == "online" and m < 0.06:
                    ax.text(xpos, m + 0.025, f"{m:.02f}",
                            ha="center", va="bottom", fontsize=8,
                            color=palette["online"])
            if "probe" in cell_means and "oracle" in cell_means:
                n_cells_total += 1
                # Treat "matches" as oracle within 0.02 of probe.
                if cell_means["oracle"] >= cell_means["probe"] - 0.02:
                    n_cells_oracle_meets += 1
        ax.set_xticks(range(len(enc_order)))
        ax.set_xticklabels(enc_order, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(cls, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if axi == 0:
            ax.set_ylabel("per-class accuracy", fontsize=11)

    print(f"[fig8] oracle matches/exceeds probe in "
          f"{n_cells_oracle_meets} of {n_cells_total} (encoder, class) cells")

    # Figure-level legend below the panels — out of the data area.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[k],
                      edgecolor="0.15", linewidth=0.7, label=label_map[k])
        for k in ["probe", "online", "oracle"]
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3,
               frameon=False, fontsize=11)
    # NO suptitle: the LaTeX \caption{} carries that text. Reserve a little
    # bottom space for the legend.
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIG_DIR / "fig8_oracle_gap_closure.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", "_preview-1.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig1_probe_vs_wm_scatter()
    fig2_perclass_bars_dk8()
    fig3_pearson_vs_horizon()
    fig4_dk8_vs_dk16()
    fig5_deadcode_dissociation()
    fig6_pearson_pooled_bootstrap()
    fig7_apples_to_apples_pooled()
    fig8_oracle_gap_closure()
    print("\nall figures written to", FIG_DIR)
