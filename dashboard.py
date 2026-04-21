"""
STVqvae Experiment Dashboard
============================
Retrofuturism-themed live dashboard for monitoring VQVAE experiments.

Tabs:
  1. Summary Comparison  — parsed experiments.md summary table with visual ranking
  2. Machine Status      — CPU / RAM / GPU utilisation + VRAM + top processes
  3. Training Progress   — TensorBoard event curves (eval_policies/logs/)
  4. Experiments Log     — live render of experiments.md

Run:
    streamlit run dashboard.py --server.port 8501

Remote access (from your laptop):
    ssh -L 8501:localhost:8501 <user>@<host>
    then open http://localhost:8501
"""

from pathlib import Path
import re
import time

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import psutil
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent
EXPERIMENTS = REPO / "experiments.md"
TB_LOG_DIR = REPO / "discrete_mbrl" / "eval_policies" / "logs"
SWEEP_LOG_DIR = REPO / "logs" / "sweep"
SWEEP_CSV = SWEEP_LOG_DIR / "sweep_results.csv"
LOG_DIRS = [
    Path("/tmp"),
    Path("/home/xiar3/experiments"),
    REPO / "wm_runs",
    REPO / "logs",
    SWEEP_LOG_DIR,          # ← sweep training logs picked up by Live Progress tab
]

# ---------------------------------------------------------------------------
# Retrofuturism colour palette
# ---------------------------------------------------------------------------
# Deep space background tones
BG_PRIMARY    = "#0a0e1a"
BG_SECONDARY  = "#111827"
BG_CARD       = "#151d2e"
BG_CARD_ALT   = "#1a2438"

# Accent & glow colours
NEON_CYAN     = "#00f0ff"
NEON_AMBER    = "#ffb347"
NEON_PINK     = "#ff2d78"
NEON_GREEN    = "#39ff85"
NEON_PURPLE   = "#c77dff"
NEON_RED      = "#ff4757"

# Text
TEXT_PRIMARY   = "#e8eaf6"
TEXT_SECONDARY = "#8892a8"
TEXT_MUTED     = "#5c6478"

# Grid & borders
GRID_COLOR     = "#1e2a42"
BORDER_COLOR   = "#253352"
BORDER_GLOW    = "#1a3a6a"

# Status
STATUS_OK     = NEON_GREEN
STATUS_WARN   = NEON_AMBER
STATUS_BAD    = NEON_RED

# Chart palette for experiments
CHART_COLORS = [NEON_CYAN, NEON_AMBER, NEON_PINK, NEON_GREEN, NEON_PURPLE,
                "#56b4e9", "#e69f00", "#cc79a7", "#009e73", "#f0e442"]


st.set_page_config(
    page_title="STVqvae // Mission Control",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Global CSS — Retrofuturism aesthetic
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ---- Import font ---- */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&family=Orbitron:wght@400;500;600;700;800;900&display=swap');

        /* ---- Global ---- */
        .stApp {{
            background: {BG_PRIMARY};
            color: {TEXT_PRIMARY};
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }}
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
            max-width: 1440px;
        }}

        /* ---- Scrollbar ---- */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: {BG_PRIMARY}; }}
        ::-webkit-scrollbar-thumb {{ background: {BORDER_GLOW}; border-radius: 3px; }}

        /* ---- Hero banner ---- */
        .hero-banner {{
            background: linear-gradient(135deg, {BG_SECONDARY} 0%, {BG_CARD} 40%, #0d1929 100%);
            border: 1px solid {BORDER_COLOR};
            border-bottom: 2px solid {NEON_CYAN}40;
            border-radius: 12px;
            padding: 1.4rem 1.8rem 1.1rem;
            margin-bottom: 1.6rem;
            position: relative;
            overflow: hidden;
        }}
        .hero-banner::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, {NEON_CYAN}, {NEON_PURPLE}, transparent);
        }}
        .hero-banner::after {{
            content: "";
            position: absolute;
            top: -60px; right: -30px;
            width: 200px; height: 200px;
            background: radial-gradient(circle, {NEON_CYAN}08 0%, transparent 70%);
            pointer-events: none;
        }}
        .hero-tag {{
            display: inline-block;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.6rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {NEON_CYAN};
            background: {NEON_CYAN}12;
            border: 1px solid {NEON_CYAN}35;
            border-radius: 4px;
            padding: 0.2rem 0.6rem;
            margin-bottom: 0.5rem;
        }}
        .hero-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            color: {TEXT_PRIMARY};
            margin: 0.3rem 0 0.15rem;
            line-height: 1.2;
        }}
        .hero-title span {{
            background: linear-gradient(90deg, {NEON_CYAN}, {NEON_PURPLE});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero-sub {{
            color: {TEXT_SECONDARY};
            font-size: 0.78rem;
            margin-top: 0.35rem;
            letter-spacing: 0.02em;
        }}
        .hero-sub strong {{
            color: {NEON_AMBER};
        }}

        /* ---- Metric cards ---- */
        .metric-card {{
            background: {BG_CARD};
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            position: relative;
            overflow: hidden;
            transition: border-color 0.2s;
        }}
        .metric-card:hover {{
            border-color: {NEON_CYAN}60;
        }}
        .metric-card::before {{
            content: "";
            position: absolute;
            left: 0; top: 0;
            width: 3px; height: 100%;
            background: linear-gradient(180deg, {NEON_CYAN} 0%, {NEON_PURPLE} 100%);
            border-radius: 3px 0 0 3px;
        }}
        .metric-card .mc-label {{
            color: {TEXT_SECONDARY};
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }}
        .metric-card .mc-value {{
            color: {TEXT_PRIMARY};
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.15;
            font-family: 'Orbitron', sans-serif;
        }}
        .metric-card .mc-badge {{
            display: inline-block;
            padding: 0.12rem 0.45rem;
            border-radius: 4px;
            font-size: 0.62rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            border: 1px solid {NEON_CYAN}40;
            background: {NEON_CYAN}15;
            color: {NEON_CYAN};
            margin-left: 0.5rem;
            vertical-align: middle;
        }}

        /* ---- Section headers ---- */
        .section-hdr {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {NEON_CYAN};
            border-bottom: 1px solid {BORDER_COLOR};
            padding-bottom: 0.5rem;
            margin-top: 1.6rem;
            margin-bottom: 1rem;
        }}

        /* ---- Separator ---- */
        .retro-sep {{
            height: 1px;
            background: linear-gradient(90deg, transparent, {NEON_CYAN}40, transparent);
            margin: 1.8rem 0;
            border: none;
        }}

        /* ---- Tab styling ---- */
        div[data-baseweb="tab-list"] {{
            gap: 0.4rem;
            border-bottom: 1px solid {BORDER_COLOR};
            padding-bottom: 0;
        }}
        div[data-baseweb="tab-list"] button {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            border-radius: 8px 8px 0 0;
            padding: 0.55rem 1rem;
            border: 1px solid {BORDER_COLOR};
            border-bottom: none;
            background: {BG_CARD};
            color: {TEXT_SECONDARY};
            transition: all 0.2s;
        }}
        div[data-baseweb="tab-list"] button:hover {{
            color: {NEON_CYAN};
            border-color: {NEON_CYAN}50;
        }}
        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: {NEON_CYAN};
            background: {BG_CARD_ALT};
            border-color: {NEON_CYAN}50;
            box-shadow: 0 -2px 12px {NEON_CYAN}15;
        }}
        /* Hide the default tab indicator line */
        div[data-baseweb="tab-highlight"] {{
            background-color: {NEON_CYAN} !important;
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] > div {{
            background: linear-gradient(180deg, {BG_SECONDARY} 0%, {BG_PRIMARY} 100%);
            border-right: 1px solid {BORDER_COLOR};
        }}
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            font-family: 'Orbitron', sans-serif;
            color: {NEON_CYAN};
            font-size: 0.8rem;
            letter-spacing: 0.06em;
        }}
        section[data-testid="stSidebar"] label {{
            color: {TEXT_SECONDARY} !important;
            font-size: 0.78rem;
        }}

        /* ---- DataFrame ---- */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            overflow: hidden;
        }}

        /* ---- Expander ---- */
        details {{
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 8px !important;
            background: {BG_CARD} !important;
        }}
        details summary {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* ---- Summary table card ---- */
        .summary-table-card {{
            background: {BG_CARD};
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1.2rem;
        }}

        /* ---- Best experiment highlight ---- */
        .best-exp {{
            background: linear-gradient(135deg, {BG_CARD} 0%, #0d2a20 100%);
            border: 1px solid {NEON_GREEN}40;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }}
        .best-exp .best-label {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {NEON_GREEN};
            margin-bottom: 0.3rem;
        }}
        .best-exp .best-name {{
            font-size: 1.3rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            font-family: 'Orbitron', sans-serif;
        }}
        .best-exp .best-stats {{
            color: {TEXT_SECONDARY};
            font-size: 0.8rem;
            margin-top: 0.3rem;
        }}
        .best-exp .best-stats span {{
            color: {NEON_GREEN};
            font-weight: 700;
        }}

        /* ---- Experiment mini cards ---- */
        .exp-row {{
            background: {BG_CARD};
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: border-color 0.2s;
        }}
        .exp-row:hover {{
            border-color: {NEON_CYAN}40;
        }}
        .exp-row .exp-name {{
            font-weight: 600;
            color: {TEXT_PRIMARY};
            font-size: 0.85rem;
            min-width: 200px;
        }}
        .exp-row .exp-metric {{
            text-align: right;
            font-size: 0.82rem;
        }}
        .exp-row .exp-metric .num {{
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            font-size: 0.95rem;
        }}

        /* ---- Status badges ---- */
        .st-done {{ color: {NEON_GREEN}; }}
        .st-running {{ color: {NEON_AMBER}; }}
        .st-pending {{ color: {TEXT_MUTED}; }}

        /* ---- Markdown content in experiment log ---- */
        .stMarkdown h1 {{
            font-family: 'Orbitron', sans-serif;
            color: {TEXT_PRIMARY};
            font-size: 1.4rem;
            letter-spacing: 0.03em;
            border-bottom: 1px solid {BORDER_COLOR};
            padding-bottom: 0.4rem;
        }}
        .stMarkdown h2 {{
            font-family: 'Orbitron', sans-serif;
            color: {NEON_CYAN};
            font-size: 1rem;
            letter-spacing: 0.04em;
            margin-top: 1.5rem;
        }}
        .stMarkdown h3 {{
            font-family: 'Orbitron', sans-serif;
            color: {NEON_AMBER};
            font-size: 0.88rem;
            letter-spacing: 0.03em;
            margin-top: 1.2rem;
        }}
        .stMarkdown h4 {{
            color: {TEXT_PRIMARY};
            font-size: 0.85rem;
            margin-top: 1rem;
        }}
        .stMarkdown table {{
            border-collapse: collapse;
            width: 100%;
        }}
        .stMarkdown table th {{
            background: {BG_CARD_ALT};
            color: {NEON_CYAN};
            font-family: 'Orbitron', sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 0.5rem 0.7rem;
            border: 1px solid {BORDER_COLOR};
        }}
        .stMarkdown table td {{
            padding: 0.45rem 0.7rem;
            border: 1px solid {BORDER_COLOR};
            font-size: 0.8rem;
            color: {TEXT_PRIMARY};
        }}
        .stMarkdown table tr:nth-child(even) {{
            background: {BG_CARD}80;
        }}
        .stMarkdown code {{
            background: {BG_CARD_ALT};
            color: {NEON_AMBER};
            padding: 0.15rem 0.35rem;
            border-radius: 4px;
            font-size: 0.82em;
        }}
        .stMarkdown pre {{
            background: {BG_CARD} !important;
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }}

        /* ---- Progress bar ---- */
        .prog-wrap {{
            background: {BG_CARD_ALT};
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            height: 10px;
            overflow: hidden;
            margin: 0.5rem 0 0.2rem;
        }}
        .prog-fill {{
            height: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, {NEON_CYAN}, {NEON_PURPLE});
            transition: width 0.4s ease;
        }}
        .prog-fill-done {{
            background: linear-gradient(90deg, {NEON_GREEN}, {NEON_CYAN});
        }}

        /* ---- Run card ---- */
        .run-card {{
            background: {BG_CARD};
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
        }}
        .run-card.active {{
            border-color: {NEON_CYAN}55;
            box-shadow: 0 0 18px {NEON_CYAN}12;
        }}
        .run-card.active::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, {NEON_CYAN}, transparent);
        }}
        .run-card.done::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, {NEON_GREEN}, transparent);
        }}
        .run-name {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.88rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            margin-bottom: 0.25rem;
        }}
        .run-meta {{
            font-size: 0.72rem;
            color: {TEXT_MUTED};
            letter-spacing: 0.02em;
            margin-bottom: 0.6rem;
        }}
        .run-stats {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            margin-top: 0.7rem;
        }}
        .run-stat {{
            display: flex;
            flex-direction: column;
        }}
        .run-stat .rs-label {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {TEXT_MUTED};
        }}
        .run-stat .rs-value {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            color: {TEXT_PRIMARY};
        }}
        .run-stat .rs-value.highlight {{
            color: {NEON_CYAN};
        }}
        .run-stat .rs-value.best {{
            color: {NEON_GREEN};
        }}
        .live-dot {{
            display: inline-block;
            width: 7px; height: 7px;
            border-radius: 50%;
            background: {NEON_GREEN};
            box-shadow: 0 0 6px {NEON_GREEN};
            margin-right: 0.4rem;
            vertical-align: middle;
        }}
        .done-dot {{
            display: inline-block;
            width: 7px; height: 7px;
            border-radius: 50%;
            background: {TEXT_MUTED};
            margin-right: 0.4rem;
            vertical-align: middle;
        }}

        /* ---- Responsive ---- */
        @media (max-width: 900px) {{
            .block-container {{
                padding-top: 0.8rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }}
            .hero-banner {{
                padding: 0.9rem 1rem;
            }}
            .hero-title {{
                font-size: 1.3rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str, badge: str | None = None) -> None:
    chip = f'<span class="mc-badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="mc-label">{label}</div>'
        f'<div class="mc-value">{value}{chip}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-hdr">{text}</div>', unsafe_allow_html=True)


def retro_sep() -> None:
    st.markdown('<div class="retro-sep"></div>', unsafe_allow_html=True)


def gauge_chart(title: str, value: float, suffix: str = "%", max_val: float = 100) -> go.Figure:
    if value < 60:
        bar_color = NEON_GREEN
    elif value < 85:
        bar_color = NEON_AMBER
    else:
        bar_color = NEON_RED

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 26, "color": TEXT_PRIMARY,
                                                 "family": "Orbitron, sans-serif"}},
            title={"text": title, "font": {"size": 12, "color": TEXT_SECONDARY,
                                            "family": "Orbitron, sans-serif"}},
            gauge={
                "axis": {
                    "range": [0, max_val],
                    "tickcolor": TEXT_MUTED,
                    "tickfont": {"color": TEXT_MUTED, "size": 9},
                },
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": BG_CARD,
                "steps": [
                    {"range": [0, 60],       "color": "rgba(57,255,133,0.06)"},
                    {"range": [60, 85],      "color": "rgba(255,179,71,0.06)"},
                    {"range": [85, max_val], "color": "rgba(255,71,87,0.06)"},
                ],
                "borderwidth": 1,
                "bordercolor": BORDER_COLOR,
            },
        )
    )
    fig.update_layout(
        height=195,
        margin=dict(t=45, b=10, l=15, r=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace"),
    )
    return fig


def smooth_series(y: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return y
    return y.rolling(window=window, min_periods=1).mean()


def markdown_sections(text: str) -> list[tuple[str, int]]:
    sections: list[tuple[str, int]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.startswith("### "):
            sections.append((line[4:].strip(), i))
    return sections


def parse_summary_table(md_text: str) -> pd.DataFrame | None:
    """Extract the '## Summary Comparison' markdown table into a DataFrame."""
    lines = md_text.splitlines()
    in_table = False
    header_line = None
    rows = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Find the Summary Comparison heading
        if stripped.startswith("## Summary Comparison"):
            in_table = True
            continue
        if in_table:
            if stripped.startswith("|") and "|" in stripped[1:]:
                # Skip separator lines like |---|---|
                if re.match(r"^\|[\s\-|:]+\|$", stripped):
                    continue
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if header_line is None:
                    header_line = cells
                else:
                    rows.append(cells)
            elif header_line and not stripped.startswith("|") and stripped not in ("", "---"):
                break  # End of table

    if not header_line or not rows:
        return None

    # Pad rows to header length
    ncols = len(header_line)
    rows = [r + [""] * (ncols - len(r)) if len(r) < ncols else r[:ncols] for r in rows]
    df = pd.DataFrame(rows, columns=header_line)
    return df


def clean_reward(val: str) -> float | None:
    """Extract a numeric reward from strings like '**0.9988** ...' or 'TBD'."""
    if not val or val.strip().upper() in ("TBD", "—", "-", ""):
        return None
    # Remove markdown bold, emojis, parenthetical text
    cleaned = re.sub(r"\*\*", "", val)
    cleaned = re.sub(r"[^\d.\-]", " ", cleaned).strip()
    parts = cleaned.split()
    if parts:
        try:
            return float(parts[0])
        except ValueError:
            return None
    return None


def status_badge(status: str) -> str:
    s = status.lower().replace("*", "")
    if "done" in s:
        return f'<span class="st-done">DONE</span>'
    elif "run" in s:
        return f'<span class="st-running">RUNNING</span>'
    else:
        return f'<span class="st-pending">PENDING</span>'


# ---------------------------------------------------------------------------
# Experiment log parsing
# ---------------------------------------------------------------------------
def _decode_log(path: Path) -> str:
    """Read a log file that may contain binary tqdm carriage-return sequences."""
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return text
    except Exception:
        return ""


def parse_tqdm_line(text: str) -> dict:
    """Extract the latest tqdm progress line: current, total, elapsed, eta, speed."""
    # tqdm lines look like:  " 14%|█▍        | 172/1221 [12:11<1:12:35,  4.17s/it]"
    matches = re.findall(
        r"(\d+)/(\d+)\s+\[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?),\s*([\d.]+)s/it\]",
        text,
    )
    if not matches:
        return {}
    cur_s, tot_s, elapsed_s, eta_s, speed_s = matches[-1]
    return {
        "current": int(cur_s),
        "total":   int(tot_s),
        "elapsed": elapsed_s,
        "eta":     eta_s,
        "speed":   float(speed_s),
        "pct":     round(int(cur_s) / max(int(tot_s), 1) * 100, 1),
    }


def parse_reward_history(text: str) -> list[tuple[int, float]]:
    """Return [(approx_update, reward), ...] from reward log lines.

    Handles three sources:
      1. Model-free custom PPO: 'New best average reward: X'
      2. World-model real-env eval callback: 'Eval: reward=X'
      3. SB3 PPO rollout table: 'ep_rew_mean | X'
    """
    results: list[tuple[int, float]] = []
    tqdm_positions: list[tuple[int, int]] = []
    for m in re.finditer(r"(\d+)/\d+\s+\[\d+:\d+", text):
        inner = re.search(r"(\d+)/\d+", m.group())
        if inner:
            tqdm_positions.append((m.start(), int(inner.group(1))))

    def _closest_update(pos: int) -> int:
        for tpos, upd in reversed(tqdm_positions):
            if tpos <= pos:
                return upd
        return 0

    # 1. Model-free: "New best average reward: X"
    for m in re.finditer(r"New best average reward:\s*([\d.]+)", text, re.IGNORECASE):
        results.append((_closest_update(m.start()), float(m.group(1))))

    # 2. World-model real-env callback: "Eval: reward=X.XXX±Y"
    for m in re.finditer(r"Eval:\s*reward=([\d.]+)", text):
        results.append((_closest_update(m.start()), float(m.group(1))))

    # 3. SB3 rollout table: "ep_rew_mean          | X"
    for m in re.finditer(r"ep_rew_mean\s*\|\s*([\d.]+)", text):
        results.append((_closest_update(m.start()), float(m.group(1))))

    # Sort by position proxy (update number) so chart is chronological
    results.sort(key=lambda x: x[0])
    return results


def parse_snapback_events(text: str) -> list[tuple[int, str]]:
    """Return [(approx_update, message), ...] for snapback trigger lines."""
    events = []
    tqdm_positions: list[tuple[int, int]] = []
    for m in re.finditer(r"(\d+)/\d+\s+\[\d+:\d+", text):
        inner = re.search(r"(\d+)/\d+", m.group())
        if inner:
            tqdm_positions.append((m.start(), int(inner.group(1))))

    for m in re.finditer(r"(snapback triggered|encoder frozen|Encoder frozen|Snapback)", text, re.IGNORECASE):
        pos = m.start()
        update = 0
        for tpos, upd in reversed(tqdm_positions):
            if tpos <= pos:
                update = upd
                break
        # grab surrounding context
        snippet = text[max(0, pos - 20): pos + 80].replace("\n", " ").strip()
        events.append((update, snippet))
    return events


def parse_run_meta(text: str) -> dict:
    """Extract run_name, total_updates, num_envs, steps_per_update from log header.

    For world-model logs (full_train_eval.py) also extracts:
      - wm: True
      - phase: 'encoder' | 'transition' | 'rl'
      - phase_progress: (current_epoch, total_epochs) or (current_ts, total_ts)
      - trans_losses: dict of step -> loss
      - enc_losses: dict
      - rl_total_steps: int
    """
    meta = {}
    m = re.search(r"--run_name\s+(\S+)", text)
    if m:
        meta["run_name"] = m.group(1)
    m = re.search(r"(\d+)/(\d+)\s+\[", text)
    if m:
        meta["total"] = int(m.group(2))
    m = re.search(r"(\d+)\s+envs\s+[×x]\s+(\d+)\s+steps\s+=\s+(\d+)\s+transitions", text)
    if m:
        meta["num_envs"]   = int(m.group(1))
        meta["steps_per"]  = int(m.group(2))
        meta["batch_size"] = int(m.group(3))

    # ---- World-model log detection ----
    if re.search(r"Step 1: Training encoder|Starting sequential training", text):
        meta["wm"] = True

        # Current phase
        if re.search(r"Step 3: Training RL|Training RL model|Starting RL", text, re.IGNORECASE):
            meta["phase"] = "rl"
        elif re.search(r"Step 2: Training transition|Constructing transition model", text, re.IGNORECASE):
            meta["phase"] = "transition"
        else:
            meta["phase"] = "encoder"

        # Encoder epoch progress  (latest "Epoch N/M:" tqdm line from encoder section)
        enc_epochs = re.findall(r"Epoch\s+(\d+)/(\d+):", text)
        if enc_epochs:
            cur, tot = enc_epochs[-1]
            meta["phase_epoch"] = (int(cur), int(tot))

        # Encoder test losses (last occurrence)
        m = re.search(
            r"Encoder test loss:\s*\{([^}]+)\}", text
        )
        if m:
            enc_losses = {}
            for kv in re.finditer(r"'([\w_]+)':\s*([\d.eE+\-]+)", m.group(1)):
                enc_losses[kv.group(1)] = float(kv.group(2))
            meta["enc_losses"] = enc_losses

        # Transition model test losses (last occurrence)
        m = re.search(
            r"Transition model test losses:\s*\{([^}]+)\}", text
        )
        if m:
            trans_losses = {}
            for kv in re.finditer(r"'([\w_]+)':\s*([\d.eE+\-]+)", m.group(1)):
                trans_losses[kv.group(1)] = float(kv.group(2))
            meta["trans_losses"] = trans_losses

        # RL timestep progress from SB3 table
        ts_matches = re.findall(r"total_timesteps\s*\|\s*(\d+)", text)
        if ts_matches:
            meta["rl_timesteps"] = int(ts_matches[-1])
        m = re.search(r"--rl_train_steps\s+(\d+)|rl_train_steps=(\d+)", text)
        if m:
            meta["rl_total_steps"] = int(m.group(1) or m.group(2))

    return meta


def discover_run_logs() -> list[Path]:
    """Find all .log files that contain tqdm training progress."""
    candidates: list[Path] = []
    for d in LOG_DIRS:
        if d.exists():
            for p in sorted(d.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True):
                if p.name == "dashboard.log":
                    continue
                candidates.append(p)
    return candidates


def is_training_log(text: str) -> bool:
    return bool(re.search(r"\d+/\d+\s+\[", text))


def is_process_running(log_path: Path) -> bool:
    """Check if a python train.py process has this log open / modified recently."""
    try:
        mtime = log_path.stat().st_mtime
        age_sec = time.time() - mtime
        return age_sec < 60  # modified within last 60 s → likely active
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------
inject_css()

# Sidebar
with st.sidebar:
    st.markdown(
        f'<div style="font-family: Orbitron, sans-serif; font-size: 0.75rem; '
        f'color: {NEON_CYAN}; letter-spacing: 0.08em; text-transform: uppercase; '
        f'margin-bottom: 0.8rem;">Control Panel</div>',
        unsafe_allow_html=True,
    )
    display_mode = st.radio("Layout", options=["Desktop", "Compact"], index=0)
    pause_refresh = st.checkbox("Pause refresh", value=False)
    refresh_sec = st.select_slider("Interval (s)", options=[3, 5, 10, 20, 30], value=5)
    top_n_proc = st.slider("Top processes", min_value=5, max_value=25, value=10)
    proc_name_filter = st.text_input("Process filter", value="")

    retro_sep()
    st.markdown(
        f'<div style="font-size: 0.68rem; color: {TEXT_MUTED}; line-height: 1.5;">'
        f'STVqvae // Mission Control<br>'
        f'MiniGrid-LavaCrossingS9N1-v0</div>',
        unsafe_allow_html=True,
    )

if not pause_refresh:
    st_autorefresh(interval=refresh_sec * 1000, key="autorefresh")

compact_mode = display_mode == "Compact"

# Hero banner
now = time.strftime("%Y-%m-%d  %H:%M:%S")
refresh_label = "PAUSED" if pause_refresh else f"{refresh_sec}s"
st.markdown(
    f'<div class="hero-banner">'
    f'<div class="hero-tag">Mission Control</div>'
    f'<div class="hero-title">STVqvae <span>Dashboard</span></div>'
    f'<div class="hero-sub">{now}  &middot;  Refresh: <strong>{refresh_label}</strong>'
    f'  &middot;  MiniGrid-LavaCrossingS9N1-v0</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_progress, tab_sweep, tab_phases, tab_rigor, tab_summary, tab_machine, tab_train, tab_exp = st.tabs([
    "Live Progress",
    "Multi-Seed Sweep",
    "Phases 6–8",
    "Full Rigor",
    "Summary Comparison",
    "Machine Status",
    "Training Progress",
    "Experiments Log",
])


# ===== TAB 0 — LIVE EXPERIMENT PROGRESS ===================================
with tab_progress:
    section_header("Live Experiment Progress")

    all_logs = discover_run_logs()
    training_logs = [(p, _decode_log(p)) for p in all_logs if is_training_log(_decode_log(p))]

    if not training_logs:
        st.info("No training logs found. Logs are scanned from `/tmp/*.log` and "
                "`/home/xiar3/experiments/*.log`.")
    else:
        # Sort: active (recently modified) first, then by mtime desc
        training_logs.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

        # Sidebar log selector
        log_names = [p.name for p, _ in training_logs]
        selected_logs = st.multiselect(
            "Select runs to display",
            options=log_names,
            default=log_names[:4],
            key="prog_log_select",
        )

        # Separate active vs finished for layout
        active_runs  = [(p, t) for p, t in training_logs if is_process_running(p) and p.name in selected_logs]
        done_runs    = [(p, t) for p, t in training_logs if not is_process_running(p) and p.name in selected_logs]

        # ---- Active runs ----
        if active_runs:
            st.markdown(
                f'<div style="font-family: Orbitron, sans-serif; font-size: 0.7rem; '
                f'color: {NEON_GREEN}; letter-spacing: 0.1em; text-transform: uppercase; '
                f'margin: 0.8rem 0 0.5rem;">Active Runs</div>',
                unsafe_allow_html=True,
            )
            for log_path, text in active_runs:
                prog   = parse_tqdm_line(text)
                meta   = parse_run_meta(text)
                hist   = parse_reward_history(text)
                snaps  = parse_snapback_events(text)
                run_name = meta.get("run_name", log_path.stem)
                pct    = prog.get("pct", 0)
                cur    = prog.get("current", 0)
                total  = prog.get("total", meta.get("total", 0))
                eta    = prog.get("eta", "—")
                speed  = prog.get("speed", 0)
                best_r = max((r for _, r in hist), default=None)
                last_r = hist[-1][1] if hist else None
                snap_txt = f"{len(snaps)} snapback(s)" if snaps else "No snapback"

                is_wm = meta.get("wm", False)

                # ---- World-model card ----
                if is_wm:
                    phase = meta.get("phase", "encoder")
                    phase_labels = {"encoder": "① Encoder", "transition": "② Transition", "rl": "③ RL Policy"}
                    phase_colors = {"encoder": NEON_PURPLE, "transition": NEON_AMBER, "rl": NEON_CYAN}
                    phase_label = phase_labels.get(phase, phase)
                    phase_color = phase_colors.get(phase, NEON_CYAN)

                    # Progress bar: epoch-based for enc/trans, timestep-based for rl
                    if phase == "rl":
                        rl_ts    = meta.get("rl_timesteps", 0)
                        rl_total = meta.get("rl_total_steps", 1)
                        fill_pct = min(round(rl_ts / max(rl_total, 1) * 100, 1), 100)
                        prog_label = f"{rl_ts:,} / {rl_total:,} timesteps ({fill_pct}%)"
                    else:
                        ep_cur, ep_tot = meta.get("phase_epoch", (cur, total or 1))
                        fill_pct = min(round(ep_cur / max(ep_tot, 1) * 100, 1), 100)
                        prog_label = f"Epoch {ep_cur} / {ep_tot} ({fill_pct}%)"

                    best_html = (f'<span class="rs-value best">{best_r:.4f}</span>'
                                 if best_r is not None else '<span class="rs-value">—</span>')
                    last_html = (f'<span class="rs-value highlight">{last_r:.4f}</span>'
                                 if last_r is not None else '<span class="rs-value">—</span>')

                    # Transition loss badges
                    trans_losses = meta.get("trans_losses", {})
                    trans_html = ""
                    for key in ["1_step_state_loss", "4_step_state_loss", "8_step_state_loss"]:
                        short = key.split("_step")[0] + "-step"
                        if key in trans_losses:
                            trans_html += (
                                f'<div class="run-stat"><span class="rs-label">{short} loss</span>'
                                f'<span class="rs-value">{trans_losses[key]:.4f}</span></div>'
                            )

                    # Encoder loss badges
                    enc_losses = meta.get("enc_losses", {})
                    enc_html = ""
                    if "quantizer_loss" in enc_losses:
                        enc_html += (
                            f'<div class="run-stat"><span class="rs-label">VQ loss</span>'
                            f'<span class="rs-value">{enc_losses["quantizer_loss"]:.4f}</span></div>'
                        )
                    if "recon_loss" in enc_losses:
                        enc_html += (
                            f'<div class="run-stat"><span class="rs-label">Recon loss</span>'
                            f'<span class="rs-value">{enc_losses["recon_loss"]:.4f}</span></div>'
                        )

                    card_html = (
                        f'<div class="run-card active">'
                        f'<div class="run-name"><span class="live-dot"></span>{run_name}'
                        f'&nbsp;&nbsp;<span style="font-size:0.65rem;color:{phase_color};'
                        f'letter-spacing:0.05em;">[{phase_label}]</span></div>'
                        f'<div class="run-meta">{log_path.name} &nbsp;·&nbsp; World-Model Pipeline</div>'
                        f'<div class="prog-wrap">'
                        f'<div class="prog-fill" style="width:{fill_pct}%;'
                        f'background:{phase_color};box-shadow:0 0 6px {phase_color}88;"></div></div>'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:0.7rem;color:{TEXT_MUTED};margin-bottom:0.5rem;">'
                        f'<span>{prog_label}</span>'
                        f'<span>ETA: {eta}</span></div>'
                        f'<div class="run-stats">'
                        f'<div class="run-stat"><span class="rs-label">Best Real Eval</span>'
                        f'{best_html}</div>'
                        f'<div class="run-stat"><span class="rs-label">Latest Real Eval</span>'
                        f'{last_html}</div>'
                        f'{enc_html}{trans_html}'
                        f'</div></div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                # ---- Standard (model-free PPO) card ----
                else:
                    fill_pct = min(pct, 100)
                    best_html  = (f'<span class="rs-value best">{best_r:.4f}</span>'
                                  if best_r is not None else '<span class="rs-value">—</span>')
                    last_html  = (f'<span class="rs-value highlight">{last_r:.4f}</span>'
                                  if last_r is not None else '<span class="rs-value">—</span>')
                    card_html = (
                        f'<div class="run-card active">'
                        f'<div class="run-name"><span class="live-dot"></span>{run_name}</div>'
                        f'<div class="run-meta">{log_path.name} &nbsp;·&nbsp; '
                        f'{meta.get("num_envs","?")} envs &times; {meta.get("steps_per","?")} steps'
                        f' &nbsp;·&nbsp; {speed:.2f}s/update</div>'
                        f'<div class="prog-wrap">'
                        f'<div class="prog-fill" style="width:{fill_pct}%"></div></div>'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:0.7rem;color:{TEXT_MUTED};margin-bottom:0.5rem;">'
                        f'<span>{cur} / {total} updates &nbsp;({pct}%)</span>'
                        f'<span>ETA: {eta}</span></div>'
                        f'<div class="run-stats">'
                        f'<div class="run-stat"><span class="rs-label">Best Reward</span>'
                        f'{best_html}</div>'
                        f'<div class="run-stat"><span class="rs-label">Latest Reward</span>'
                        f'{last_html}</div>'
                        f'<div class="run-stat"><span class="rs-label">Snapback</span>'
                        f'<span class="rs-value">{snap_txt}</span></div>'
                        f'</div></div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                # Reward history chart (inline, compact) — shown for both types when data exists
                if hist:
                    updates_h = [u for u, _ in hist]
                    rewards_h = [r for _, r in hist]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=updates_h, y=rewards_h,
                        mode="lines+markers",
                        line=dict(color=NEON_CYAN, width=2),
                        marker=dict(size=5, color=NEON_CYAN),
                        name="Real-env reward" if is_wm else "Best reward",
                        fill="tozeroy",
                        fillcolor=f"rgba(0,240,255,0.05)",
                    ))
                    for snap_u, _ in snaps:
                        fig.add_vline(
                            x=snap_u,
                            line=dict(color=NEON_AMBER, dash="dash", width=1),
                            annotation_text="snapback",
                            annotation_font=dict(color=NEON_AMBER, size=10),
                        )
                    fig.update_layout(
                        height=200,
                        margin=dict(t=10, b=25, l=35, r=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="JetBrains Mono, monospace", color=TEXT_SECONDARY),
                        xaxis=dict(title="Update", gridcolor=GRID_COLOR, zeroline=False,
                                   range=[0, total or 1]),
                        yaxis=dict(title="Reward", gridcolor=GRID_COLOR, zeroline=False,
                                   range=[0, 1.05]),
                        showlegend=False,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"active_chart_{log_path.stem}")

        retro_sep()

        # ---- Finished runs ----
        if done_runs:
            st.markdown(
                f'<div style="font-family: Orbitron, sans-serif; font-size: 0.7rem; '
                f'color: {TEXT_SECONDARY}; letter-spacing: 0.1em; text-transform: uppercase; '
                f'margin-bottom: 0.5rem;">Completed Runs</div>',
                unsafe_allow_html=True,
            )
            for log_path, text in done_runs:
                prog   = parse_tqdm_line(text)
                meta   = parse_run_meta(text)
                hist   = parse_reward_history(text)
                snaps  = parse_snapback_events(text)
                run_name = meta.get("run_name", log_path.stem)
                cur    = prog.get("current", 0)
                total  = prog.get("total", meta.get("total", 0))
                best_r = max((r for _, r in hist), default=None)

                # Final avg from log
                final_m = re.search(r"Final \d+-episode average:\s*([\d.]+)", text)
                final_r = float(final_m.group(1)) if final_m else None
                overall_m = re.search(r"Overall average reward:\s*([\d.]+)", text)
                overall_r = float(overall_m.group(1)) if overall_m else None

                snap_txt = f"{len(snaps)} snapback(s)" if snaps else "No snapback"
                pct = round(cur / max(total, 1) * 100, 1) if total else 100

                st.markdown(
                    f'<div class="run-card done">'
                    f'<div class="run-name"><span class="done-dot"></span>{run_name}'
                    f'<span style="font-size:0.65rem; color:{NEON_GREEN}; '
                    f'margin-left:0.6rem; font-family: JetBrains Mono;">DONE</span></div>'
                    f'<div class="run-meta">{log_path.name}</div>'
                    f'<div class="prog-wrap"><div class="prog-fill prog-fill-done" style="width:100%"></div></div>'
                    f'<div style="font-size:0.7rem; color:{TEXT_MUTED}; margin-bottom:0.5rem;">'
                    f'{cur} / {total} updates  ·  100%</div>'
                    f'<div class="run-stats">'
                    + (f'<div class="run-stat"><span class="rs-label">Peak Reward</span>'
                       f'<span class="rs-value best">{best_r:.4f}</span></div>' if best_r else '')
                    + (f'<div class="run-stat"><span class="rs-label">Final Avg</span>'
                       f'<span class="rs-value">{final_r:.4f}</span></div>' if final_r else '')
                    + (f'<div class="run-stat"><span class="rs-label">Overall Avg</span>'
                       f'<span class="rs-value">{overall_r:.4f}</span></div>' if overall_r else '')
                    + f'<div class="run-stat"><span class="rs-label">Snapback</span>'
                      f'<span class="rs-value">{snap_txt}</span></div>'
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )

                # Collapsed reward history chart
                if hist:
                    with st.expander(f"Reward history — {run_name}", expanded=False):
                        updates_h = [u for u, _ in hist]
                        rewards_h = [r for _, r in hist]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=updates_h, y=rewards_h,
                            mode="lines+markers",
                            line=dict(color=NEON_GREEN, width=2),
                            marker=dict(size=5, color=NEON_GREEN),
                            fill="tozeroy",
                            fillcolor="rgba(57,255,133,0.05)",
                        ))
                        for snap_u, _ in snaps:
                            fig.add_vline(
                                x=snap_u,
                                line=dict(color=NEON_AMBER, dash="dash", width=1),
                                annotation_text="snapback",
                                annotation_font=dict(color=NEON_AMBER, size=10),
                            )
                        fig.update_layout(
                            height=220,
                            margin=dict(t=10, b=25, l=35, r=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="JetBrains Mono, monospace", color=TEXT_SECONDARY),
                            xaxis=dict(title="Update", gridcolor=GRID_COLOR, zeroline=False),
                            yaxis=dict(title="Reward", gridcolor=GRID_COLOR, zeroline=False,
                                       range=[0, 1.05]),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"done_chart_{log_path.stem}")

        # ---- Multi-run comparison chart ----
        all_selected = active_runs + done_runs
        if len(all_selected) >= 2:
            retro_sep()
            section_header("Reward Comparison — All Selected Runs")
            fig_cmp = go.Figure()
            for idx, (log_path, text) in enumerate(all_selected):
                meta = parse_run_meta(text)
                hist = parse_reward_history(text)
                if not hist:
                    continue
                run_name = meta.get("run_name", log_path.stem)
                updates_h = [u for u, _ in hist]
                rewards_h = [r for _, r in hist]
                fig_cmp.add_trace(go.Scatter(
                    x=updates_h, y=rewards_h,
                    mode="lines+markers",
                    name=run_name,
                    line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2),
                    marker=dict(size=5),
                ))
            fig_cmp.update_layout(
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono, monospace", color=TEXT_SECONDARY),
                xaxis=dict(title="Update", gridcolor=GRID_COLOR, zeroline=False),
                yaxis=dict(title="Best Reward", gridcolor=GRID_COLOR, zeroline=False,
                           range=[0, 1.05]),
                legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10, b=35, l=40, r=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig_cmp, use_container_width=True, key="multi_run_comparison")


# ===== TAB 1 — MULTI-SEED SWEEP ===========================================
with tab_sweep:
    section_header("Multi-Seed Sweep — DoorKey-8x8")

    # ── Constants ──
    SWEEP_ENCODERS = ["v2", "v5", "v6", "v9"]
    SWEEP_SEEDS    = [1, 2, 3]
    TOTAL_RUNS     = len(SWEEP_ENCODERS) * len(SWEEP_SEEDS)

    # ── Load CSV results ──
    sweep_df = None
    if SWEEP_CSV.exists():
        try:
            sweep_df = pd.read_csv(SWEEP_CSV)
        except Exception:
            sweep_df = None

    # ── Build status dict  {(enc, seed): "done"|"active"|"pending"} ──
    def _sweep_run_name(enc, seed):
        return f"sweep_doorkey8_{enc}_s{seed}"

    def _sweep_log(enc, seed):
        return SWEEP_LOG_DIR / f"{_sweep_run_name(enc, seed)}.log"

    status_map = {}
    for enc in SWEEP_ENCODERS:
        for seed in SWEEP_SEEDS:
            log_p = _sweep_log(enc, seed)
            in_csv = (sweep_df is not None and
                      len(sweep_df[(sweep_df["encoder"] == enc) & (sweep_df["seed"] == seed)]) > 0)
            if in_csv:
                status_map[(enc, seed)] = "done"
            elif log_p.exists() and is_process_running(log_p):
                status_map[(enc, seed)] = "active"
            elif log_p.exists():
                status_map[(enc, seed)] = "done"   # finished but CSV not yet updated
            else:
                status_map[(enc, seed)] = "pending"

    n_done    = sum(1 for s in status_map.values() if s == "done")
    n_active  = sum(1 for s in status_map.values() if s == "active")
    n_pending = sum(1 for s in status_map.values() if s == "pending")

    # ── KPI row ──
    kc1, kc2, kc3, kc4 = st.columns(4, gap="medium")
    with kc1:
        metric_card("Total Runs", str(TOTAL_RUNS))
    with kc2:
        metric_card("Completed", str(n_done), badge=f"{n_done}/{TOTAL_RUNS}")
    with kc3:
        metric_card("Active", str(n_active))
    with kc4:
        # ETA: remaining runs × avg time (assume ~2h/run)
        remaining = n_active + n_pending
        eta_h = remaining * 2.0
        metric_card("Remaining", f"{remaining} runs", badge=f"~{eta_h:.0f}h")

    # ── Progress bar ──
    overall_pct = round(n_done / TOTAL_RUNS * 100, 1)
    st.markdown(
        f'<div style="margin: 0.8rem 0 0.3rem; font-size:0.72rem; '
        f'color:{TEXT_MUTED}; letter-spacing:0.05em;">Overall sweep progress</div>'
        f'<div class="prog-wrap">'
        f'<div class="prog-fill" style="width:{overall_pct}%"></div></div>'
        f'<div style="font-size:0.7rem; color:{TEXT_MUTED}; '
        f'margin-top:0.25rem;">{overall_pct}% ({n_done}/{TOTAL_RUNS} runs)</div>',
        unsafe_allow_html=True,
    )

    retro_sep()

    # ── 12-run grid (encoders as columns, seeds as rows) ──
    section_header("Run Status Grid")

    dot_html = {
        "done":    f'<span style="color:{NEON_GREEN};">●</span>',
        "active":  f'<span style="color:{NEON_AMBER}; animation:blink 1s step-end infinite;">●</span>',
        "pending": f'<span style="color:{TEXT_MUTED};">○</span>',
    }

    # Header row
    hdr_cols = st.columns([1] + [2] * len(SWEEP_ENCODERS), gap="small")
    hdr_cols[0].markdown(
        f'<div style="font-size:0.68rem;color:{TEXT_MUTED};letter-spacing:0.06em;">SEED</div>',
        unsafe_allow_html=True)
    for i, enc in enumerate(SWEEP_ENCODERS):
        hdr_cols[i + 1].markdown(
            f'<div style="font-family:Orbitron,sans-serif;font-size:0.75rem;'
            f'color:{NEON_CYAN};text-align:center;">{enc.upper()}</div>',
            unsafe_allow_html=True)

    for seed in SWEEP_SEEDS:
        row_cols = st.columns([1] + [2] * len(SWEEP_ENCODERS), gap="small")
        row_cols[0].markdown(
            f'<div style="font-size:0.8rem;color:{TEXT_SECONDARY};padding-top:0.25rem;">s{seed}</div>',
            unsafe_allow_html=True)

        for i, enc in enumerate(SWEEP_ENCODERS):
            st_key = status_map.get((enc, seed), "pending")
            log_p  = _sweep_log(enc, seed)

            # If log exists, extract best reward
            best_r_str = "—"
            if log_p.exists():
                try:
                    txt = _decode_log(log_p)
                    hist = parse_reward_history(txt)
                    if hist:
                        best_r_str = f"{max(r for _, r in hist):.4f}"
                    # If still running, show progress too
                    if st_key == "active":
                        prog = parse_tqdm_line(txt)
                        pct_v = prog.get("pct", 0)
                        eta_v = prog.get("eta", "?")
                        extra = f"<br><span style='font-size:0.62rem;color:{TEXT_MUTED};'>{pct_v}% · ETA {eta_v}</span>"
                    else:
                        extra = ""
                except Exception:
                    extra = ""
            else:
                extra = ""

            dot = dot_html[st_key]
            color = NEON_GREEN if st_key == "done" else (NEON_AMBER if st_key == "active" else TEXT_MUTED)
            row_cols[i + 1].markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER_COLOR};'
                f'border-radius:8px;padding:0.5rem 0.6rem;text-align:center;">'
                f'<div style="font-size:0.9rem;">{dot}</div>'
                f'<div style="font-family:Orbitron,sans-serif;font-size:0.78rem;'
                f'color:{color};margin-top:0.1rem;">{best_r_str}</div>'
                f'{extra}</div>',
                unsafe_allow_html=True,
            )

    retro_sep()

    # ── Active run detail ──
    active_pairs = [(enc, seed) for (enc, seed), s in status_map.items() if s == "active"]
    if active_pairs:
        section_header("Active Run Detail")
        for enc, seed in active_pairs:
            log_p = _sweep_log(enc, seed)
            txt   = _decode_log(log_p)
            prog  = parse_tqdm_line(txt)
            hist  = parse_reward_history(txt)
            snaps = parse_snapback_events(txt)
            cur   = prog.get("current", 0)
            total = prog.get("total", 1221)
            pct   = prog.get("pct", 0)
            eta   = prog.get("eta", "—")
            speed = prog.get("speed", 0)
            best_r = max((r for _, r in hist), default=None)
            last_r = hist[-1][1] if hist else None
            run_name = _sweep_run_name(enc, seed)

            best_html = (f'<span class="rs-value best">{best_r:.4f}</span>'
                         if best_r is not None else '<span class="rs-value">—</span>')
            last_html = (f'<span class="rs-value highlight">{last_r:.4f}</span>'
                         if last_r is not None else '<span class="rs-value">—</span>')

            st.markdown(
                f'<div class="run-card active">'
                f'<div class="run-name"><span class="live-dot"></span>{run_name}</div>'
                f'<div class="run-meta">Encoder: {enc.upper()} &nbsp;·&nbsp; Seed: {seed}'
                f' &nbsp;·&nbsp; {speed:.2f}s/update</div>'
                f'<div class="prog-wrap"><div class="prog-fill" style="width:{pct}%"></div></div>'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.7rem;color:{TEXT_MUTED};margin-bottom:0.5rem;">'
                f'<span>{cur} / {total} updates &nbsp;({pct}%)</span>'
                f'<span>ETA: {eta}</span></div>'
                f'<div class="run-stats">'
                f'<div class="run-stat"><span class="rs-label">Best Reward</span>{best_html}</div>'
                f'<div class="run-stat"><span class="rs-label">Latest</span>{last_html}</div>'
                f'<div class="run-stat"><span class="rs-label">Snapbacks</span>'
                f'<span class="rs-value">{len(snaps)}</span></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            if hist:
                updates_h = [u for u, _ in hist]
                rewards_h = [r for _, r in hist]
                fig_act = go.Figure()
                fig_act.add_trace(go.Scatter(
                    x=updates_h, y=rewards_h,
                    mode="lines+markers",
                    line=dict(color=NEON_CYAN, width=2),
                    marker=dict(size=5, color=NEON_CYAN),
                    fill="tozeroy",
                    fillcolor="rgba(0,240,255,0.05)",
                    name="best reward",
                ))
                fig_act.update_layout(
                    height=200,
                    margin=dict(t=8, b=25, l=35, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="JetBrains Mono, monospace", color=TEXT_SECONDARY),
                    xaxis=dict(title="Update", gridcolor=GRID_COLOR, zeroline=False,
                               range=[0, total or 1221]),
                    yaxis=dict(title="Reward", gridcolor=GRID_COLOR, zeroline=False,
                               range=[0, 1.05]),
                    showlegend=False,
                )
                st.plotly_chart(fig_act, use_container_width=True,
                                key=f"sweep_active_{enc}_s{seed}")

        retro_sep()

    # ── Completed results table ──
    done_logs = [(enc, seed) for (enc, seed), s in status_map.items() if s == "done"]
    if done_logs:
        section_header("Completed Run Results")

        # Build results from logs (more up-to-date than CSV during active sweep)
        result_rows = []
        for enc, seed in done_logs:
            log_p = _sweep_log(enc, seed)
            row = {"Encoder": enc.upper(), "Seed": seed, "Run": _sweep_run_name(enc, seed)}
            if log_p.exists():
                txt = _decode_log(log_p)
                hist = parse_reward_history(txt)
                row["Best Reward"] = round(max((r for _, r in hist), default=0), 4)
                final_m = re.search(r"Final \d+-episode average:\s*([\d.]+)", txt)
                row["Final Avg"] = round(float(final_m.group(1)), 4) if final_m else None
                overall_m = re.search(r"Overall average reward:\s*([\d.]+)", txt)
                row["Overall Avg"] = round(float(overall_m.group(1)), 4) if overall_m else None
            # Probe results from CSV if available
            if sweep_df is not None:
                match = sweep_df[(sweep_df["encoder"] == enc) & (sweep_df["seed"] == seed)]
                if not match.empty:
                    r = match.iloc[0]
                    for col in ["probe_overall", "probe_wall", "probe_floor",
                                "probe_door", "probe_key", "probe_goal", "probe_agent"]:
                        if col in r and str(r[col]) not in ("NA", "", "nan"):
                            row[col.replace("probe_", "probe ").title()] = r[col]
            result_rows.append(row)

        if result_rows:
            res_df = pd.DataFrame(result_rows)
            st.dataframe(res_df, use_container_width=True, hide_index=True,
                         height=min(500, len(res_df) * 38 + 60))

        retro_sep()

        # ── Reward comparison chart for completed runs ──
        if len(done_logs) >= 2:
            section_header("Completed Runs — Reward Comparison")
            fig_cmp = go.Figure()
            color_map = {"v2": NEON_AMBER, "v5": NEON_CYAN,
                         "v6": NEON_GREEN, "v9": NEON_PURPLE}
            dash_map  = {1: "solid", 2: "dash", 3: "dot"}
            for enc, seed in done_logs:
                log_p = _sweep_log(enc, seed)
                if not log_p.exists():
                    continue
                txt  = _decode_log(log_p)
                hist = parse_reward_history(txt)
                if not hist:
                    continue
                updates_h = [u for u, _ in hist]
                rewards_h = [r for _, r in hist]
                fig_cmp.add_trace(go.Scatter(
                    x=updates_h, y=rewards_h,
                    mode="lines",
                    name=f"{enc.upper()} s{seed}",
                    line=dict(
                        color=color_map.get(enc, NEON_CYAN),
                        width=2,
                        dash=dash_map.get(seed, "solid"),
                    ),
                ))
            fig_cmp.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono, monospace", color=TEXT_SECONDARY),
                xaxis=dict(title="Update", gridcolor=GRID_COLOR, zeroline=False),
                yaxis=dict(title="Best Reward", gridcolor=GRID_COLOR, zeroline=False,
                           range=[0, 1.05]),
                legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                            groupclick="toggleitem"),
                margin=dict(t=10, b=35, l=40, r=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig_cmp, use_container_width=True, key="sweep_reward_comparison")

        # ── Per-encoder summary (mean ± std across seeds) ──
        if sweep_df is not None and not sweep_df.empty and n_done >= 2:
            retro_sep()
            section_header("Per-Encoder Summary (mean ± std across seeds)")
            try:
                num_cols = [c for c in sweep_df.columns
                            if c not in ("encoder", "seed") and
                            pd.to_numeric(sweep_df[c], errors="coerce").notna().any()]
                agg = sweep_df.copy()
                for c in num_cols:
                    agg[c] = pd.to_numeric(agg[c], errors="coerce")
                grp = agg.groupby("encoder")[num_cols].agg(["mean", "std"]).round(4)
                st.dataframe(grp, use_container_width=True)
            except Exception:
                pass

    elif n_pending == TOTAL_RUNS:
        st.info("Sweep not started yet. All runs are pending.")
    else:
        st.info("No runs completed yet — check back soon.")


# ===== TAB 2 — PHASES 6–8 LIVE PROGRESS ==================================
with tab_phases:
    section_header("Phases 6–8: Scaling, VAE Baseline & Crafter")

    DK16_LOG_DIR  = REPO / "logs" / "sweep_dk16"
    DK16_CSV      = DK16_LOG_DIR / "sweep_dk16_results.csv"
    WM_OUT_DIR    = REPO / "logs" / "wm_analysis"
    CB_OUT_DIR    = REPO / "logs" / "codebook_analysis"
    VAE_LOG       = REPO / "logs" / "vae_baseline_doorkey_v6enc.log"

    # ── helper: extract best reward from a training log ──
    def _best_from_log(log_path):
        import re as _re
        if not Path(log_path).exists():
            return None
        txt = Path(log_path).read_text(errors="replace")
        hits = _re.findall(r"New best average reward:\s+([\d.]+)", txt)
        return float(hits[-1]) if hits else None

    def _iter_from_log(log_path, total):
        """Return (current_iter, total) by counting tqdm-style lines."""
        import re as _re
        if not Path(log_path).exists():
            return 0, total
        txt = Path(log_path).read_text(errors="replace")
        hits = _re.findall(r"\|\s*(\d+)/(\d+)\s*\[", txt)
        if hits:
            cur, tot = hits[-1]
            return int(cur), int(tot)
        return 0, total

    # ── PHASE 6 — DoorKey-16x16 multi-seed ──────────────────────────────────
    st.markdown(f"### 🔷 Phase 6 — DoorKey-16×16 Multi-Seed Sweep")

    dk16_df = None
    if DK16_CSV.exists():
        try:
            dk16_df = pd.read_csv(DK16_CSV)
        except Exception:
            dk16_df = None

    SEEDS_16 = [1, 2, 3]
    completed_seeds = set(dk16_df["seed"].tolist()) if dk16_df is not None else set()

    # KPI row
    k1, k2, k3 = st.columns(3)
    with k1:
        n_done = len(completed_seeds)
        st.metric("Seeds complete", f"{n_done}/3",
                  delta="training" if n_done < 3 else "done")
    with k2:
        if dk16_df is not None and len(dk16_df):
            avg_best = dk16_df["best_reward"].replace("NA", float("nan")).astype(float).mean()
            st.metric("Mean best reward", f"{avg_best:.4f}")
        else:
            st.metric("Mean best reward", "—")
    with k3:
        # find actively training seed
        import subprocess, re as _re
        try:
            ps_out = subprocess.check_output(
                ["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
            dk16_procs = [l for l in ps_out.splitlines()
                          if "train.py" in l and "DoorKey-16x16" in l]
            if dk16_procs:
                m = _re.search(r"sweep_doorkey16_v6_s(\d+)", dk16_procs[0])
                active_seed = m.group(1) if m else "?"
                st.metric("Active seed", f"s{active_seed} training")
            else:
                st.metric("Active seed", "idle / done")
        except Exception:
            st.metric("Active seed", "—")

    # Seed progress bars
    for seed in SEEDS_16:
        log_path = DK16_LOG_DIR / f"sweep_doorkey16_v6_s{seed}.log"
        best = _best_from_log(log_path)
        cur_iter, tot_iter = _iter_from_log(log_path, 1954)

        if seed in completed_seeds:
            row = dk16_df[dk16_df["seed"] == seed].iloc[0]
            label = f"**Seed {seed}** ✅  best={row['best_reward']}  final={row['final_reward']}  avg={row['overall_avg']}"
            pct = 1.0
        elif log_path.exists() and cur_iter > 0:
            label = f"**Seed {seed}** 🔄  best so far={best or '—'}  ({cur_iter}/{tot_iter} batches)"
            pct = cur_iter / tot_iter
        else:
            label = f"**Seed {seed}** ⏳ queued"
            pct = 0.0

        st.markdown(label)
        st.progress(pct)

    # Completed results table
    if dk16_df is not None and len(dk16_df):
        st.markdown("**Completed results:**")
        disp = dk16_df[["seed","best_reward","final_reward","overall_avg",
                         "probe_goal","probe_door","probe_wall"]].copy()
        st.dataframe(disp, use_container_width=True)

    # Codebook highlight for each completed seed
    cb_jsons = sorted(DK16_LOG_DIR.glob("*_codebook.json"))
    if cb_jsons:
        st.markdown("**Codebook allocation (16×16):**")
        cb_rows = []
        for jf in cb_jsons:
            try:
                import json as _json
                d = _json.loads(jf.read_text())
                seed_m = _re.search(r"_s(\d+)_", jf.name)
                seed_n = seed_m.group(1) if seed_m else "?"
                cpc = d.get("class_code_count", {})
                cb_rows.append({
                    "seed": seed_n,
                    "active": d.get("n_active", "?"),
                    "dead": d.get("n_dead", "?"),
                    "dead%": f"{d.get('dead_fraction',0)*100:.0f}%",
                    "empty_codes": cpc.get("1", 0),
                    "wall_codes":  cpc.get("2", 0),
                    "goal_codes":  cpc.get("8", 0),
                    "door_codes":  cpc.get("4", 0),
                    "key_codes":   cpc.get("5", 0),
                    "agent_codes": cpc.get("10", 0),
                })
            except Exception:
                pass
        if cb_rows:
            st.dataframe(pd.DataFrame(cb_rows), use_container_width=True)

    st.divider()

    # ── PHASE 7 — VAE baseline ───────────────────────────────────────────────
    st.markdown("### 🟣 Phase 7 — Continuous VAE Baseline (DoorKey-8×8)")

    vae_best  = _best_from_log(VAE_LOG)
    vae_cur, vae_tot = _iter_from_log(VAE_LOG, 1221)
    vae_pct   = vae_cur / vae_tot if vae_tot > 0 else 0.0

    v7a, v7b = st.columns(2)
    with v7a:
        st.metric("Best reward so far", f"{vae_best:.4f}" if vae_best else "—")
    with v7b:
        st.metric("Progress", f"{vae_cur}/{vae_tot} batches ({vae_pct*100:.0f}%)")

    if VAE_LOG.exists():
        st.progress(vae_pct)
        # show last few lines of log (strip tqdm noise)
        import re as _re
        lines = VAE_LOG.read_text(errors="replace").splitlines()
        clean = [l for l in lines if l.strip() and "%|" not in l and
                 "FutureWarn" not in l and "Gym has" not in l][-10:]
        st.code("\n".join(clean), language=None)
    else:
        st.info("VAE baseline not yet started.")

    st.divider()

    # ── PHASE 8 — Crafter codebook analysis ─────────────────────────────────
    st.markdown("### 🟡 Phase 8 — Crafter Codebook Analysis")

    CRAFTER_MODELS = ["v6enc_original", "v6enc_fix", "v6enc_cal", "v6enc_cal2"]
    crafter_jsons  = {p.stem.replace("crafter_",""): p
                      for p in CB_OUT_DIR.glob("crafter_*.json")}
    crafter_done   = [m for m in CRAFTER_MODELS if m in crafter_jsons]

    p8a, p8b = st.columns(2)
    with p8a:
        st.metric("Models analyzed", f"{len(crafter_done)}/{len(CRAFTER_MODELS)}")
    with p8b:
        remaining = [m for m in CRAFTER_MODELS if m not in crafter_jsons]
        st.metric("Remaining", ", ".join(remaining) if remaining else "All done ✅")

    # Results table
    if crafter_jsons:
        import json as _json
        crafter_rows = []
        CRAFTER_CLASSES = {
            '0':'invalid','1':'water','2':'grass','3':'stone','4':'path',
            '5':'sand','6':'tree','7':'lava','8':'coal','9':'iron',
            '10':'diamond','11':'table','12':'furnace','13':'plant',
            '14':'fence','15':'player','16':'cow','17':'zombie','18':'skeleton'
        }
        for model_name in CRAFTER_MODELS:
            if model_name not in crafter_jsons:
                crafter_rows.append({"model": model_name, "status": "⏳ pending"})
                continue
            try:
                d = _json.loads(crafter_jsons[model_name].read_text())
                cb = d.get("codebook_size", "?")
                n_act = d.get("n_active", "?")
                n_dead = d.get("n_dead", "?")
                dead_frac = d.get("dead_fraction", 0)
                avg_purity = d.get("avg_purity", 0)
                cpc = d.get("class_code_count", {})
                # interesting classes
                row = {
                    "model": model_name,
                    "status": "✅",
                    "cb_size": cb,
                    "active": n_act,
                    "dead%": f"{dead_frac*100:.0f}%",
                    "purity": f"{avg_purity*100:.1f}%",
                }
                for cid, cname in [("15","player"),("16","cow"),("17","zombie"),
                                    ("6","tree"),("2","grass"),("3","stone")]:
                    row[cname] = cpc.get(cid, 0)
                crafter_rows.append(row)
            except Exception as e:
                crafter_rows.append({"model": model_name, "status": f"❌ {e}"})
        st.dataframe(pd.DataFrame(crafter_rows), use_container_width=True)

    # WM analysis summary (Phase 5, shown here for reference)
    wm_jsons = sorted(WM_OUT_DIR.glob("*.json"))
    if wm_jsons:
        st.markdown("**Phase 5 WM Accuracy recap (all 8 models):**")
        import json as _json
        wm_rows = []
        for jf in wm_jsons:
            try:
                d = _json.loads(jf.read_text())
                scatter = {s["name"]: s for s in d.get("scatter_data", [])}
                wm_rows.append({
                    "model": jf.stem,
                    "Pearson r": f"{d.get('pearson_r', 0):.3f}",
                    "goal_probe": f"{scatter.get('goal',{}).get('probe_acc',0)*100:.0f}%",
                    "goal_WM":    f"{scatter.get('goal',{}).get('wm_acc',0)*100:.0f}%",
                    "door_WM":    f"{scatter.get('door',{}).get('wm_acc',0)*100:.0f}%",
                    "wall_WM":    f"{scatter.get('wall',{}).get('wm_acc',0)*100:.0f}%",
                })
            except Exception:
                pass
        if wm_rows:
            wm_df = pd.DataFrame(wm_rows).sort_values("Pearson r", ascending=False)
            st.dataframe(wm_df, use_container_width=True)


# ===== TAB 3b — FULL RIGOR SWEEP ==========================================
with tab_rigor:
    import json as _json_r, re as _re_r, subprocess as _sub_r

    section_header("Full-Rigor Sweep — DK16 Resolution + DK8 Multi-Seed")

    RIGOR_DIR   = REPO / "logs" / "full_rigor"
    RIGOR_CSV   = RIGOR_DIR / "full_rigor_results.csv"
    CURVES_PNG  = REPO / "logs" / "rl_transfer_curves.png"

    def _rigor_best(log_path):
        if not Path(log_path).exists():
            return None
        txt = Path(log_path).read_text(errors="replace")
        hits = _re_r.findall(r"New best average reward:\s+([\d.]+)", txt)
        return float(hits[-1]) if hits else None

    def _rigor_iter(log_path, total=1954):
        if not Path(log_path).exists():
            return 0, total
        txt = Path(log_path).read_text(errors="replace")
        hits = _re_r.findall(r"\|\s*(\d+)/(\d+)\s*\[", txt)
        if hits:
            cur, tot = hits[-1]
            return int(cur), int(tot)
        return 0, total

    def _active_run():
        """Return run_name of currently training process (grep ps)."""
        try:
            ps = _sub_r.check_output(["ps", "aux"], text=True, stderr=_sub_r.DEVNULL)
            for line in ps.splitlines():
                if "train.py" in line:
                    m = _re_r.search(r"--run_name\s+(\S+)", line)
                    if m:
                        return m.group(1)
        except Exception:
            pass
        return None

    active = _active_run()

    # ── KPI strip ────────────────────────────────────────────────────────────
    rigor_df = None
    if RIGOR_CSV.exists():
        try:
            rigor_df = pd.read_csv(RIGOR_CSV)
        except Exception:
            pass

    n_total  = 10   # A1, A2, + 8 DK8 runs
    n_done   = len(rigor_df) if rigor_df is not None else 0
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Runs complete", f"{n_done}/{n_total}")
    with k2:
        st.metric("Currently running", active or "idle")
    with k3:
        hrs_left = max(0, (n_total - n_done) * 4)
        st.metric("Est. hours remaining", f"~{hrs_left} hr")

    st.divider()

    # ── (A) DK-16 Resolution ─────────────────────────────────────────────────
    st.markdown("### 🔷 (A) DK-16 Resolution — Aggressive Restart vs Smaller Codebook")

    A_RUNS = [
        ("rigor_dk16_cb512_thr2", "A1: cb=512, thr=2.0 (aggressive restart)", 1954),
        ("rigor_dk16_cb256_thr1", "A2: cb=256,  thr=1.0 (smaller codebook)",  1954),
    ]
    for run_name, label, total in A_RUNS:
        tlog = RIGOR_DIR / f"{run_name}_train.log"
        plog = RIGOR_DIR / f"{run_name}_probe.log"
        cbjson = RIGOR_DIR / f"{run_name}_codebook.json"
        best = _rigor_best(tlog)
        cur, tot = _rigor_iter(tlog, total)
        pct = cur / tot if tot > 0 else 0.0

        if rigor_df is not None and run_name in rigor_df.get("variant", pd.Series()).values:
            row = rigor_df[rigor_df["variant"] == run_name].iloc[0]
            status = f"✅ best={row.get('best_reward','?')}  door={row.get('probe_door','?')}%  key={row.get('probe_key','?')}%"
        elif tlog.exists() and cur > 0:
            status = f"🔄 {cur}/{tot} iters  best so far={best or '—'}"
        else:
            status = "⏳ queued"

        st.markdown(f"**{label}** — {status}")
        st.progress(pct)

        # Show codebook result if done
        if cbjson.exists():
            try:
                d = _json_r.loads(cbjson.read_text())
                ca, cd, df_ = d.get("n_active","?"), d.get("n_dead","?"), d.get("dead_fraction",0)
                st.caption(f"  Codebook: {ca} active / {cd} dead ({df_*100:.0f}% dead)")
            except Exception:
                pass

        # Show probe if done
        if plog.exists():
            try:
                txt = plog.read_text(errors="replace")
                door_m = _re_r.search(r"door.*?([\d.]+)%", txt)
                key_m  = _re_r.search(r" key .*?([\d.]+)%", txt)
                goal_m = _re_r.search(r"goal.*?([\d.]+)%", txt)
                parts = []
                if door_m: parts.append(f"door={door_m.group(1)}%")
                if key_m:  parts.append(f"key={key_m.group(1)}%")
                if goal_m: parts.append(f"goal={goal_m.group(1)}%")
                if parts:
                    st.caption("  Probe: " + "  ".join(parts))
            except Exception:
                pass

    st.divider()

    # ── (B) DK-8 Multi-seed ───────────────────────────────────────────────────
    st.markdown("### 🟢 (B) DK-8 Multi-Seed — Error Bars for Paper (seeds 1 & 2)")

    B_VARIANTS = [
        ("v2",   "v2 (baseline VQ encoder)"),
        ("v5dc", "v5+dc (dead-code restart)"),
        ("v6",   "v6 (RGB+coord+trunk)"),
        ("vae",  "VAE (no VQ bottleneck)"),
    ]
    B_SEEDS = [1, 2]
    B_TOTAL = 1221  # 5M steps / 4096 batch ≈ 1221

    for var_key, var_label in B_VARIANTS:
        st.markdown(f"**{var_label}**")
        cols = st.columns(len(B_SEEDS))
        for ci, seed in enumerate(B_SEEDS):
            run_name = f"rigor_dk8_{var_key}_s{seed}"
            tlog = RIGOR_DIR / f"{run_name}_train.log"
            best = _rigor_best(tlog)
            cur, tot = _rigor_iter(tlog, B_TOTAL)
            pct = cur / tot if tot > 0 else 0.0
            with cols[ci]:
                if rigor_df is not None and run_name in rigor_df.get("variant", pd.Series()).values:
                    row = rigor_df[rigor_df["variant"] == run_name].iloc[0]
                    label_s = f"s{seed} ✅ best={row.get('best_reward','?')}"
                elif tlog.exists() and cur > 0:
                    label_s = f"s{seed} 🔄 {cur}/{tot} ({pct*100:.0f}%)"
                else:
                    label_s = f"s{seed} ⏳ queued"
                st.caption(label_s)
                st.progress(pct)

    st.divider()

    # ── Results table ─────────────────────────────────────────────────────────
    if rigor_df is not None and not rigor_df.empty:
        st.markdown("### 📊 Accumulated Results")
        st.dataframe(rigor_df, use_container_width=True)

        # Seed-wise best-reward summary by variant
        b_rows = rigor_df[rigor_df["group"] == "B_dk8"].copy()
        if not b_rows.empty:
            b_rows["best_reward"] = pd.to_numeric(b_rows["best_reward"], errors="coerce")
            st.markdown("**DK-8 multi-seed summary (mean ± std):**")
            summ = b_rows.groupby("variant")["best_reward"].agg(["mean","std","count"])
            summ.columns = ["mean_best","std_best","n_seeds"]
            st.dataframe(summ.reset_index(), use_container_width=True)

    st.divider()

    # ── RL Transfer Curve plot ─────────────────────────────────────────────────
    st.markdown("### 📈 RL Transfer Curves (DK-8)")
    if CURVES_PNG.exists():
        st.image(str(CURVES_PNG), caption="Running best reward vs PPO iteration — v2 / v5+dc / v6 / VAE")
        st.caption("v2 log will appear once its first multi-seed run completes.")
    else:
        st.info("Plot not yet generated (`logs/rl_transfer_curves.png`).")

    # ── Live launcher tail ────────────────────────────────────────────────────
    LAUNCHER_LOG = REPO / "logs" / "full_rigor_launcher.out"
    if LAUNCHER_LOG.exists():
        st.markdown("### 🖥 Launcher output (tail)")
        lines = LAUNCHER_LOG.read_text(errors="replace").splitlines()
        clean = [l for l in lines if l.strip() and "%|" not in l][-20:]
        st.code("\n".join(clean), language=None)


# ===== TAB 3 — SUMMARY COMPARISON =========================================
with tab_summary:
    if not EXPERIMENTS.exists():
        st.warning(f"`{EXPERIMENTS}` not found.")
    else:
        content = EXPERIMENTS.read_text()
        df_summary = parse_summary_table(content)

        if df_summary is None or df_summary.empty:
            st.info("No summary table found in experiments.md. "
                    "Ensure there is a `## Summary Comparison` section with a markdown table.")
        else:
            # ---- Parse numeric reward columns ----
            best_col = "Best Reward" if "Best Reward" in df_summary.columns else None
            final_col = "Final Avg" if "Final Avg" in df_summary.columns else None

            if best_col:
                df_summary["_best_num"] = df_summary[best_col].apply(clean_reward)
            if final_col:
                df_summary["_final_num"] = df_summary[final_col].apply(clean_reward)

            # ---- Highlight best experiment ----
            if best_col and "_best_num" in df_summary.columns:
                valid = df_summary.dropna(subset=["_best_num"])
                if not valid.empty:
                    best_idx = valid["_best_num"].idxmax()
                    best_row = df_summary.loc[best_idx]
                    best_name = re.sub(r"\*", "", best_row.get("Experiment", ""))
                    best_val = best_row["_best_num"]
                    final_val = best_row.get("_final_num", None)
                    final_str = f"  &middot;  Final: <span>{final_val:.4f}</span>" if final_val else ""
                    key_change = re.sub(r"\*", "", best_row.get("Key Change", ""))

                    st.markdown(
                        f'<div class="best-exp">'
                        f'<div class="best-label">Current Leader</div>'
                        f'<div class="best-name">{best_name}</div>'
                        f'<div class="best-stats">'
                        f'Best: <span>{best_val:.4f}</span>{final_str}'
                        f'<br>{key_change}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("", unsafe_allow_html=True)  # spacer

            # ---- Bar chart: Best Reward comparison ----
            if best_col and "_best_num" in df_summary.columns:
                section_header("Best Reward by Experiment")

                chart_df = df_summary.dropna(subset=["_best_num"]).copy()
                chart_df["_name"] = chart_df["Experiment"].str.replace(r"\*", "", regex=True)
                chart_df = chart_df.sort_values("_best_num", ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=chart_df["_name"],
                    x=chart_df["_best_num"],
                    orientation="h",
                    marker=dict(
                        color=chart_df["_best_num"],
                        colorscale=[
                            [0, NEON_RED],
                            [0.3, NEON_AMBER],
                            [0.7, NEON_CYAN],
                            [1.0, NEON_GREEN],
                        ],
                        line=dict(width=0),
                    ),
                    text=chart_df["_best_num"].apply(lambda v: f"{v:.4f}"),
                    textposition="outside",
                    textfont=dict(color=TEXT_SECONDARY, size=10,
                                  family="JetBrains Mono, monospace"),
                ))
                fig.update_layout(
                    height=max(280, len(chart_df) * 32 + 80),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="JetBrains Mono, monospace", color=TEXT_SECONDARY),
                    xaxis=dict(
                        title="Best Reward",
                        gridcolor=GRID_COLOR,
                        zeroline=False,
                        range=[0, max(chart_df["_best_num"].max() * 1.15, 1.05)],
                        title_font=dict(size=11),
                    ),
                    yaxis=dict(
                        gridcolor=GRID_COLOR,
                        tickfont=dict(size=10),
                    ),
                    margin=dict(t=10, b=30, l=10, r=50),
                    bargap=0.35,
                )
                st.plotly_chart(fig, use_container_width=True, key="summary_best_reward_bar")

            # ---- Final Avg comparison (if available) ----
            if final_col and "_final_num" in df_summary.columns:
                chart_df2 = df_summary.dropna(subset=["_final_num"]).copy()
                chart_df2 = chart_df2[chart_df2["_final_num"] > 0]
                if not chart_df2.empty:
                    section_header("Final Average Reward (non-zero only)")

                    chart_df2["_name"] = chart_df2["Experiment"].str.replace(r"\*", "", regex=True)
                    chart_df2 = chart_df2.sort_values("_final_num", ascending=True)

                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        y=chart_df2["_name"],
                        x=chart_df2["_final_num"],
                        orientation="h",
                        marker=dict(
                            color=chart_df2["_final_num"],
                            colorscale=[
                                [0, NEON_AMBER],
                                [0.5, NEON_CYAN],
                                [1.0, NEON_GREEN],
                            ],
                            line=dict(width=0),
                        ),
                        text=chart_df2["_final_num"].apply(lambda v: f"{v:.4f}"),
                        textposition="outside",
                        textfont=dict(color=TEXT_SECONDARY, size=10,
                                      family="JetBrains Mono, monospace"),
                    ))
                    fig2.update_layout(
                        height=max(200, len(chart_df2) * 32 + 80),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="JetBrains Mono, monospace", color=TEXT_SECONDARY),
                        xaxis=dict(
                            title="Final Avg Reward",
                            gridcolor=GRID_COLOR,
                            zeroline=False,
                            range=[0, max(chart_df2["_final_num"].max() * 1.15, 1.05)],
                            title_font=dict(size=11),
                        ),
                        yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=10)),
                        margin=dict(t=10, b=30, l=10, r=50),
                        bargap=0.35,
                    )
                    st.plotly_chart(fig2, use_container_width=True, key="summary_final_reward_bar")

            # ---- Scatter: Best vs Final ----
            if (best_col and final_col
                    and "_best_num" in df_summary.columns
                    and "_final_num" in df_summary.columns):
                scatter_df = df_summary.dropna(subset=["_best_num", "_final_num"]).copy()
                scatter_df = scatter_df[(scatter_df["_best_num"] > 0) | (scatter_df["_final_num"] > 0)]
                if len(scatter_df) >= 2:
                    section_header("Peak vs Final Reward")

                    scatter_df["_name"] = scatter_df["Experiment"].str.replace(r"\*", "", regex=True)

                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=scatter_df["_best_num"],
                        y=scatter_df["_final_num"],
                        mode="markers+text",
                        marker=dict(
                            size=12,
                            color=scatter_df["_final_num"],
                            colorscale=[[0, NEON_RED], [0.5, NEON_AMBER], [1, NEON_GREEN]],
                            line=dict(width=1, color=BORDER_COLOR),
                        ),
                        text=scatter_df["_name"],
                        textposition="top center",
                        textfont=dict(size=9, color=TEXT_SECONDARY,
                                      family="JetBrains Mono, monospace"),
                    ))
                    # Diagonal: final = best (ideal)
                    mx = max(scatter_df["_best_num"].max(), scatter_df["_final_num"].max()) * 1.1
                    fig3.add_trace(go.Scatter(
                        x=[0, mx], y=[0, mx],
                        mode="lines",
                        line=dict(color=TEXT_MUTED, dash="dot", width=1),
                        showlegend=False,
                    ))
                    fig3.update_layout(
                        height=380,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="JetBrains Mono, monospace", color=TEXT_SECONDARY),
                        xaxis=dict(title="Peak Reward", gridcolor=GRID_COLOR, zeroline=False),
                        yaxis=dict(title="Final Avg", gridcolor=GRID_COLOR, zeroline=False),
                        margin=dict(t=15, b=35, l=35, r=15),
                        showlegend=False,
                    )
                    st.plotly_chart(fig3, use_container_width=True, key="summary_scatter_best_vs_final")

            # ---- Full table ----
            retro_sep()
            section_header("Full Summary Table")

            # Display dataframe without internal columns
            display_df = df_summary.drop(
                columns=[c for c in df_summary.columns if c.startswith("_")],
                errors="ignore",
            )
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=min(600, len(display_df) * 38 + 60),
            )


# ===== TAB 2 — MACHINE STATUS ==============================================
with tab_machine:
    cpu_pct = psutil.cpu_percent(interval=0.35)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # KPI cards row
    kpi_cols = st.columns(2 if compact_mode else 4, gap="medium")
    kpi_data = [
        ("CPU", f"{cpu_pct:.1f}%"),
        ("RAM", f"{ram.percent:.1f}%"),
        ("Disk /", f"{disk.percent:.1f}%"),
        ("Cores", f"{psutil.cpu_count(logical=False)}P / {psutil.cpu_count()}T"),
    ]
    for idx, (label, value) in enumerate(kpi_data):
        with kpi_cols[idx % len(kpi_cols)]:
            metric_card(label, value)

    st.markdown("")  # spacer

    # Gauges
    section_header("System Gauges")
    g1, g2, g3 = st.columns(3 if not compact_mode else 1, gap="medium")

    with g1:
        st.plotly_chart(gauge_chart("CPU", cpu_pct), use_container_width=True, key="machine_cpu_gauge")
        st.caption(f"Machine-wide compute utilisation")
    with g2:
        st.plotly_chart(gauge_chart("RAM", ram.percent), use_container_width=True, key="machine_ram_gauge")
        st.caption(f"{ram.used / 1024**3:.1f} GB / {ram.total / 1024**3:.1f} GB")
    with g3:
        st.plotly_chart(gauge_chart("Disk", disk.percent), use_container_width=True, key="machine_disk_gauge")
        st.caption(f"{disk.used / 1024**3:.0f} GB / {disk.total / 1024**3:.0f} GB")

    retro_sep()

    # GPU
    section_header("GPU Telemetry")
    try:
        from pynvml import (
            nvmlInit, nvmlShutdown, nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex, nvmlDeviceGetName,
            nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
        )
        nvmlInit()
        n_gpus = nvmlDeviceGetCount()
        if n_gpus == 0:
            st.info("No GPUs detected.")
        else:
            gpu_cols = st.columns(min(3, n_gpus) if not compact_mode else 1, gap="medium")
            for i in range(n_gpus):
                handle = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(handle)
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                vram_pct = mem.used / mem.total * 100

                with gpu_cols[i % len(gpu_cols)]:
                    metric_card(f"GPU {i}", str(name), badge=f"{util.gpu}% util")
                    st.markdown("")
                    st.plotly_chart(gauge_chart(f"Compute", util.gpu), use_container_width=True, key=f"gpu_compute_{i}")
                    st.plotly_chart(gauge_chart(f"VRAM", vram_pct), use_container_width=True, key=f"gpu_vram_{i}")
                    st.caption(
                        f"VRAM {mem.used / 1024**3:.1f} / {mem.total / 1024**3:.1f} GB  ·  "
                        f"Mem BW {util.memory}%"
                    )
        nvmlShutdown()
    except Exception as e:
        st.warning(f"GPU monitoring unavailable: {e}")

    retro_sep()

    # Processes
    section_header("Active Processes")
    try:
        rows = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status"]):
            try:
                rows.append({
                    "PID": p.info["pid"],
                    "Name": p.info["name"],
                    "CPU %": p.info["cpu_percent"],
                    "RAM MB": round(p.info["memory_info"].rss / 1024**2, 1),
                    "Status": p.info["status"],
                })
            except Exception:
                pass

        proc_df = pd.DataFrame(rows)
        if not proc_df.empty:
            if proc_name_filter.strip():
                proc_df = proc_df[
                    proc_df["Name"].str.contains(
                        re.escape(proc_name_filter.strip()), case=False, regex=True
                    )
                ]
            proc_df = (
                proc_df.sort_values("CPU %", ascending=False)
                .head(top_n_proc)
                .reset_index(drop=True)
            )
        st.dataframe(proc_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Process list unavailable: {e}")


# ===== TAB 3 — TRAINING PROGRESS ==========================================
with tab_train:
    section_header("TensorBoard Curves")

    if not TB_LOG_DIR.exists():
        st.info(f"No log directory found at `{TB_LOG_DIR}`")
    else:
        try:
            from tbparse import SummaryReader

            run_dirs = sorted(
                [p for p in TB_LOG_DIR.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
            )
            if not run_dirs:
                run_dirs = [TB_LOG_DIR]

            run_names = [r.name for r in run_dirs]

            c1, c2, c3 = st.columns([2, 2, 1], gap="medium")
            with c1:
                selected_runs = st.multiselect("Runs", options=run_names, default=run_names)
            with c2:
                metric_filter = st.text_input("Metric filter", value="reward|loss")
            with c3:
                smooth_w = st.slider("Smoothing", min_value=1, max_value=50, value=5)

            all_scalars: list[pd.DataFrame] = []
            for run_dir in run_dirs:
                if run_dir.name not in selected_runs:
                    continue
                try:
                    df = SummaryReader(str(run_dir), pivot=False).scalars
                    if df is not None and not df.empty:
                        df["run"] = run_dir.name
                        all_scalars.append(df)
                except Exception:
                    pass

            if not all_scalars:
                st.info("No scalar data found for selected runs.")
            else:
                combined = pd.concat(all_scalars, ignore_index=True)
                tags = sorted(combined["tag"].unique())
                if metric_filter.strip():
                    regex = re.compile(metric_filter, re.IGNORECASE)
                    tags = [t for t in tags if regex.search(t)]

                default_n = 4 if compact_mode else 8
                selected_tags = st.multiselect("Metrics", options=tags, default=tags[:default_n])
                if not selected_tags:
                    st.info("Select at least one metric.")
                else:
                    for tag_idx, tag in enumerate(selected_tags):
                        subset = combined[combined["tag"] == tag]
                        fig = go.Figure()
                        for r_idx, run in enumerate(subset["run"].unique()):
                            run_data = subset[subset["run"] == run].sort_values("step")
                            y = smooth_series(run_data["value"], smooth_w)
                            fig.add_trace(go.Scatter(
                                x=run_data["step"],
                                y=y,
                                mode="lines",
                                name=run,
                                line={"width": 2, "color": CHART_COLORS[r_idx % len(CHART_COLORS)]},
                            ))
                        fig.update_layout(
                            title=dict(text=tag, font=dict(size=13, color=TEXT_PRIMARY,
                                                            family="Orbitron, sans-serif")),
                            xaxis_title="Step",
                            yaxis_title="Value",
                            height=260 if compact_mode else 340,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="JetBrains Mono, monospace", color=TEXT_SECONDARY),
                            legend=dict(
                                font=dict(size=10),
                                bgcolor="rgba(0,0,0,0)",
                            ),
                            xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
                            yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
                            margin=dict(t=45, b=25, l=25, r=15),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig, use_container_width=True, key="tb_scalars_chart")

                    with st.expander("Raw scalar data"):
                        st.dataframe(
                            combined[combined["tag"].isin(selected_tags)],
                            use_container_width=True,
                        )

        except ImportError:
            st.error("tbparse not installed. Run: `pip install tbparse`")
        except Exception as e:
            st.error(f"Error reading TensorBoard logs: {e}")


# ===== TAB 4 — EXPERIMENTS LOG =============================================
with tab_exp:
    section_header("Experiments Log")

    if not EXPERIMENTS.exists():
        st.warning(f"`{EXPERIMENTS}` not found.")
    else:
        mtime = EXPERIMENTS.stat().st_mtime
        content = EXPERIMENTS.read_text()

        c1, c2 = st.columns([3, 1] if not compact_mode else [1, 1])
        with c1:
            st.caption(
                f"Last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}"
            )
        with c2:
            raw_view = st.toggle("Raw markdown", value=False)

        sections = markdown_sections(content)
        if sections:
            labels = [f"L{line}: {title}" for title, line in sections]
            selected = st.selectbox("Jump to section", options=["(None)"] + labels)
            if selected != "(None)":
                line_no = int(selected.split(":", 1)[0][1:])
                lines = content.splitlines()
                start = max(0, line_no - 1)
                end = min(len(lines), start + 120)
                with st.expander("Section preview", expanded=False):
                    st.code("\n".join(lines[start:end]), language="markdown")

        retro_sep()

        if raw_view:
            st.code(content, language="markdown")
        else:
            st.markdown(content, unsafe_allow_html=False)
