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
LOG_DIRS = [Path("/tmp"), Path("/home/xiar3/experiments"), REPO / "wm_runs", REPO / "logs"]

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
tab_progress, tab_summary, tab_machine, tab_train, tab_exp = st.tabs([
    "Live Progress",
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


# ===== TAB 1 — SUMMARY COMPARISON =========================================
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
