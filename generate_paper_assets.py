import json
import os
import re
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "paper_assets")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

LAVA_LOG = os.path.join(ROOT, "wm_runs", "eval_semantic_compare_v2.log")
DOORKEY_LOG = os.path.join(ROOT, "wm_runs", "eval_semantic_doorkey_v8enc.log")
TRANSFER_GAP_JSON = os.path.join(ROOT, "wm_runs", "vae_wm_rl_v9_transfer_gap", "results.json")


LAVA_ORDER = ["exp31_wm_only", "exp33_sem_aux", "exp34_sem_v2"]
DOORKEY_ORDER = [
    "exp32_dk_wm_only",
    "exp36_dk_v5enc",
    "exp37_dk_v5_prevq",
    "exp38_dk_v6_goal",
    "exp39_dk_v7_multiscale",
    "exp40_dk_v8_gated",
]

LAVA_CLASS_ORDER = ["empty", "wall", "goal", "lava", "agent", "overall"]
DOORKEY_CLASS_ORDER = ["empty", "wall", "door", "key", "goal", "agent", "overall"]

DISPLAY_NAMES = {
    "exp31_wm_only": "WM-only",
    "exp33_sem_aux": "Semantic aux",
    "exp34_sem_v2": "Semantic aux v2",
    "exp32_dk_wm_only": "WM-only",
    "exp36_dk_v5enc": "v5enc",
    "exp37_dk_v5_prevq": "v5+prevq",
    "exp38_dk_v6_goal": "v6 goal-aware",
    "exp39_dk_v7_multiscale": "v7 multiscale",
    "exp40_dk_v8_gated": "v8 gated",
}

PAPER_COLORS = {
    "gray": "#7a7a7a",
    "blue": "#2457a7",
    "teal": "#1f8a8a",
    "orange": "#c96a1b",
    "green": "#3f7d20",
    "red": "#b33c2f",
}


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "font.size": 11,
            "grid.color": "#cfcfcf",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.45,
            "savefig.bbox": "tight",
        }
    )


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def parse_probe_log(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    pattern = re.compile(
        r"\[(?P<model>[^\]]+)\]\s+overall accuracy:\s+(?P<overall>[\d.]+)%"
        r"(?P<body>.*?)(?=\n\s*\[[^\]]+\]\s+sample|\n=+\n\s*Model:|\Z)",
        re.S,
    )
    class_pattern = re.compile(r"^\s*(\w+)\s+\(id=\s*\d+\):\s+([\d.]+)%", re.M)

    results = OrderedDict()
    for match in pattern.finditer(text):
        model = match.group("model").strip()
        overall = float(match.group("overall"))
        body = match.group("body")
        metrics = OrderedDict()
        for class_name, value in class_pattern.findall(body):
            metrics[class_name] = float(value)
        metrics["overall"] = overall
        results[model] = metrics
    return results


def parse_transfer_gap(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    real_reward = data["real_env_eval"]["mean_reward"]
    real_len = data["real_env_eval"]["mean_episode_length"]
    horizons = []
    wm_rewards = []
    wm_lengths = []
    for horizon in sorted(data["world_model_eval"], key=lambda x: int(x)):
        horizons.append(int(horizon))
        wm_rewards.append(data["world_model_eval"][horizon]["mean_reward"])
        wm_lengths.append(data["world_model_eval"][horizon]["mean_episode_length"])
    return {
        "real_reward": real_reward,
        "real_length": real_len,
        "horizons": horizons,
        "wm_rewards": wm_rewards,
        "wm_lengths": wm_lengths,
    }


def make_grouped_bar(metrics, model_order, class_order, title, out_path, ylabel):
    model_labels = [DISPLAY_NAMES.get(m, m) for m in model_order]
    x = np.arange(len(class_order))
    width = 0.8 / len(model_order)

    colors = [
        PAPER_COLORS["gray"],
        PAPER_COLORS["blue"],
        PAPER_COLORS["teal"],
        PAPER_COLORS["orange"],
        PAPER_COLORS["green"],
        PAPER_COLORS["red"],
    ]
    plt.figure(figsize=(10.6, 4.9))
    for idx, model in enumerate(model_order):
        values = [metrics[model].get(cls, np.nan) for cls in class_order]
        offset = (idx - (len(model_order) - 1) / 2) * width
        plt.bar(
            x + offset,
            values,
            width=width,
            label=model_labels[idx],
            color=colors[idx],
            edgecolor="white",
            linewidth=0.6,
        )

    plt.xticks(x, [c.capitalize() for c in class_order], rotation=0)
    plt.ylim(0, 105)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")
    plt.gca().set_axisbelow(True)
    plt.legend(frameon=False, ncol=min(3, len(model_order)), loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_overall_bar(lava_metrics, doorkey_metrics):
    lava_models = [DISPLAY_NAMES.get(m, m) for m in LAVA_ORDER]
    dk_models = [DISPLAY_NAMES.get(m, m) for m in DOORKEY_ORDER]
    lava_vals = [lava_metrics[m]["overall"] for m in LAVA_ORDER]
    dk_vals = [doorkey_metrics[m]["overall"] for m in DOORKEY_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    axes[0].bar(lava_models, lava_vals, color=[PAPER_COLORS["gray"], PAPER_COLORS["blue"], PAPER_COLORS["teal"]], edgecolor="white", linewidth=0.7)
    axes[0].set_title("LavaCrossing overall probe accuracy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(axis="y")
    axes[0].set_axisbelow(True)

    colors = [
        PAPER_COLORS["gray"],
        PAPER_COLORS["blue"],
        PAPER_COLORS["teal"],
        PAPER_COLORS["orange"],
        PAPER_COLORS["green"],
        PAPER_COLORS["red"],
    ]
    axes[1].bar(dk_models, dk_vals, color=colors[: len(dk_models)], edgecolor="white", linewidth=0.7)
    axes[1].set_title("DoorKey overall probe accuracy")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y")
    axes[1].set_axisbelow(True)

    for ax in axes:
        ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "overall_probe_accuracy.png"), dpi=200)
    plt.close(fig)


def make_transfer_gap_plot(data):
    horizons = data["horizons"]
    wm_rewards = data["wm_rewards"]
    real_reward = data["real_reward"]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(
        horizons,
        wm_rewards,
        marker="o",
        markersize=6.5,
        linewidth=2.2,
        color=PAPER_COLORS["orange"],
        label="Imagined world-model reward",
    )
    ax.axhline(
        real_reward,
        linestyle="--",
        linewidth=2.0,
        color=PAPER_COLORS["blue"],
        label="Real-environment reward",
    )
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Mean reward")
    ax.set_title("LavaCrossing transfer gap: imagined vs real reward")
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "transfer_gap_reward.png"), dpi=200)
    plt.close(fig)


def write_markdown_table(path, headers, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")


def write_latex_table(path, caption, label, headers, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")


def fmt(v):
    return f"{v:.1f}"


def make_probe_tables(metrics, model_order, class_order, stem, caption, label):
    headers = ["Model"] + [c.capitalize() for c in class_order]
    rows = []
    for model in model_order:
        row = [DISPLAY_NAMES.get(model, model)] + [fmt(metrics[model][c]) for c in class_order]
        rows.append(row)

    write_markdown_table(os.path.join(TABLE_DIR, f"{stem}.md"), headers, rows)
    write_latex_table(os.path.join(TABLE_DIR, f"{stem}.tex"), caption, label, headers, rows)


def make_transfer_gap_table(data):
    headers = ["Setting", "Mean reward", "Mean length"]
    rows = [["Real environment", f"{data['real_reward']:.4f}", f"{data['real_length']:.1f}"]]
    for horizon, reward, length in zip(data["horizons"], data["wm_rewards"], data["wm_lengths"]):
        rows.append([f"World model (h={horizon})", f"{reward:.4f}", f"{length:.1f}"])

    write_markdown_table(os.path.join(TABLE_DIR, "transfer_gap.md"), headers, rows)
    write_latex_table(
        os.path.join(TABLE_DIR, "transfer_gap.tex"),
        "LavaCrossing transfer gap between real-environment evaluation and imagined world-model rollouts.",
        "tab:transfer_gap",
        headers,
        rows,
    )


def write_readme():
    text = """# Paper Assets

Generated assets for the paper draft.

## Figures
- `figures/lavacrossing_probe_accuracy.png`: per-class semantic probe accuracy on LavaCrossing
- `figures/doorkey_probe_accuracy.png`: per-class semantic probe accuracy on DoorKey
- `figures/overall_probe_accuracy.png`: overall semantic probe accuracy summary
- `figures/transfer_gap_reward.png`: imagined vs real reward across rollout horizons

## Tables
- `tables/lavacrossing_probe.{md,tex}`
- `tables/doorkey_probe.{md,tex}`
- `tables/transfer_gap.{md,tex}`
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


def main():
    configure_style()
    ensure_dirs()

    lava_metrics = parse_probe_log(LAVA_LOG)
    doorkey_metrics = parse_probe_log(DOORKEY_LOG)
    transfer_gap = parse_transfer_gap(TRANSFER_GAP_JSON)

    make_grouped_bar(
        lava_metrics,
        LAVA_ORDER,
        LAVA_CLASS_ORDER,
        "LavaCrossing semantic probe accuracy",
        os.path.join(FIG_DIR, "lavacrossing_probe_accuracy.png"),
        "Accuracy (%)",
    )
    make_grouped_bar(
        doorkey_metrics,
        DOORKEY_ORDER,
        DOORKEY_CLASS_ORDER,
        "DoorKey semantic probe accuracy",
        os.path.join(FIG_DIR, "doorkey_probe_accuracy.png"),
        "Accuracy (%)",
    )
    make_overall_bar(lava_metrics, doorkey_metrics)
    make_transfer_gap_plot(transfer_gap)

    make_probe_tables(
        lava_metrics,
        LAVA_ORDER,
        LAVA_CLASS_ORDER,
        "lavacrossing_probe",
        "Semantic probe accuracy on LavaCrossing. Standard world-model and generic semantic variants fail to recover the goal.",
        "tab:lava_probe",
    )
    make_probe_tables(
        doorkey_metrics,
        DOORKEY_ORDER,
        DOORKEY_CLASS_ORDER,
        "doorkey_probe",
        "Semantic probe accuracy on DoorKey. The goal-aware variant strongly improves recoverability of the sparse goal object.",
        "tab:doorkey_probe",
    )
    make_transfer_gap_table(transfer_gap)
    write_readme()


if __name__ == "__main__":
    main()
