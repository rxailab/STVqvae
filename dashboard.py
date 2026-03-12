"""
STVqvae Live Dashboard
======================
Tabs:
  1. Machine Status  — CPU / RAM / GPU utilisation + VRAM
  2. Training Progress — TensorBoard event curves (eval_policies/logs/)
  3. Experiments      — live render of experiments.md

Run:
    streamlit run dashboard.py --server.port 8501

Remote access (from your laptop):
    ssh -L 8501:localhost:8501 <user>@<host>
    then open http://localhost:8501
"""

from pathlib import Path
import time

import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── paths ──────────────────────────────────────────────────────────────────────
REPO          = Path(__file__).parent
EXPERIMENTS   = REPO / "experiments.md"
TB_LOG_DIR    = REPO / "discrete_mbrl" / "eval_policies" / "logs"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STVqvae Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# auto-refresh every 5 s
st_autorefresh(interval=5_000, key="autorefresh")

st.title("STVqvae Live Dashboard")
st.caption(f"Auto-refreshes every 5 s · {time.strftime('%Y-%m-%d %H:%M:%S')}")

tab_machine, tab_train, tab_exp = st.tabs(
    ["⚙️ Machine Status", "📈 Training Progress", "📋 Experiments"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MACHINE STATUS
# ══════════════════════════════════════════════════════════════════════════════
with tab_machine:
    st.subheader("System Resources")

    # ── CPU / RAM ─────────────────────────────────────────────────────────────
    cpu_pct   = psutil.cpu_percent(interval=0.5)
    ram       = psutil.virtual_memory()
    ram_pct   = ram.percent
    ram_used  = ram.used  / 1024**3
    ram_total = ram.total / 1024**3

    # disk
    disk = psutil.disk_usage("/")
    disk_pct = disk.percent

    col1, col2, col3 = st.columns(3)

    def gauge(title, value, suffix="%", max_val=100, color_thresholds=(60, 85)):
        lo, hi = color_thresholds
        color = "#2ecc71" if value < lo else ("#f39c12" if value < hi else "#e74c3c")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 28}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, max_val]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0, lo],       "color": "#1a1a2e"},
                    {"range": [lo, hi],      "color": "#16213e"},
                    {"range": [hi, max_val], "color": "#0f3460"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        ))
        fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        return fig

    with col1:
        st.plotly_chart(gauge("CPU", cpu_pct), use_container_width=True)
        st.caption(f"{psutil.cpu_count(logical=False)} physical / "
                   f"{psutil.cpu_count()} logical cores")

    with col2:
        st.plotly_chart(gauge("RAM", ram_pct), use_container_width=True)
        st.caption(f"{ram_used:.1f} GB / {ram_total:.1f} GB used")

    with col3:
        st.plotly_chart(gauge("Disk (/)", disk_pct), use_container_width=True)
        st.caption(f"{disk.used/1024**3:.0f} GB / {disk.total/1024**3:.0f} GB used")

    # ── GPU ───────────────────────────────────────────────────────────────────
    st.subheader("GPU(s)")
    try:
        from pynvml import (
            nvmlInit, nvmlShutdown,
            nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName, nvmlDeviceGetUtilizationRates,
            nvmlDeviceGetMemoryInfo,
        )
        nvmlInit()
        n_gpus = nvmlDeviceGetCount()
        gpu_cols = st.columns(min(n_gpus, 4))
        for i in range(n_gpus):
            handle  = nvmlDeviceGetHandleByIndex(i)
            name    = nvmlDeviceGetName(handle)
            util    = nvmlDeviceGetUtilizationRates(handle)
            mem     = nvmlDeviceGetMemoryInfo(handle)
            vram_pct = mem.used / mem.total * 100
            with gpu_cols[i % len(gpu_cols)]:
                st.markdown(f"**GPU {i}: {name}**")
                st.plotly_chart(
                    gauge(f"GPU {i} Compute", util.gpu),
                    use_container_width=True,
                )
                st.plotly_chart(
                    gauge(f"GPU {i} VRAM",
                          round(vram_pct, 1),
                          suffix="%",
                          max_val=100),
                    use_container_width=True,
                )
                st.caption(
                    f"{mem.used/1024**3:.1f} GB / {mem.total/1024**3:.1f} GB VRAM  ·  "
                    f"MEM util {util.memory}%"
                )
        nvmlShutdown()
    except Exception as e:
        st.warning(f"GPU monitoring unavailable: {e}")

    # ── per-process top-5 ─────────────────────────────────────────────────────
    st.subheader("Top Processes by CPU")
    try:
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status"]):
            try:
                procs.append({
                    "PID":    p.info["pid"],
                    "Name":   p.info["name"],
                    "CPU %":  p.info["cpu_percent"],
                    "RAM MB": round(p.info["memory_info"].rss / 1024**2, 1),
                    "Status": p.info["status"],
                })
            except Exception:
                pass
        df_proc = (
            pd.DataFrame(procs)
            .sort_values("CPU %", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        st.dataframe(df_proc, use_container_width=True)
    except Exception as e:
        st.warning(f"Process list unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING PROGRESS
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("TensorBoard Training Curves")

    if not TB_LOG_DIR.exists():
        st.info(f"No log directory found at `{TB_LOG_DIR}`")
    else:
        try:
            from tbparse import SummaryReader

            # discover all run sub-dirs
            run_dirs = sorted(
                [p for p in TB_LOG_DIR.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
            )
            if not run_dirs:
                # flat event files at the root
                run_dirs = [TB_LOG_DIR]

            run_names = [r.name for r in run_dirs]
            selected_runs = st.multiselect(
                "Select runs to display",
                options=run_names,
                default=run_names,
            )

            all_scalars: list[pd.DataFrame] = []
            for run_dir in run_dirs:
                if run_dir.name not in selected_runs:
                    continue
                try:
                    reader = SummaryReader(str(run_dir), pivot=False)
                    df = reader.scalars
                    if df is not None and not df.empty:
                        df["run"] = run_dir.name
                        all_scalars.append(df)
                except Exception:
                    pass

            if not all_scalars:
                st.info("No scalar data found in the log directories yet.")
            else:
                combined = pd.concat(all_scalars, ignore_index=True)

                tags = sorted(combined["tag"].unique())
                selected_tags = st.multiselect(
                    "Select metrics to plot",
                    options=tags,
                    default=[t for t in tags if "reward" in t.lower() or "loss" in t.lower()][:6] or tags[:6],
                )

                for tag in selected_tags:
                    subset = combined[combined["tag"] == tag]
                    fig = go.Figure()
                    for run in subset["run"].unique():
                        run_data = subset[subset["run"] == run].sort_values("step")
                        fig.add_trace(go.Scatter(
                            x=run_data["step"],
                            y=run_data["value"],
                            mode="lines",
                            name=run,
                            line={"width": 2},
                        ))
                    fig.update_layout(
                        title=tag,
                        xaxis_title="Step",
                        yaxis_title="Value",
                        legend_title="Run",
                        height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(20,20,40,0.8)",
                        font_color="white",
                        xaxis={"gridcolor": "#333"},
                        yaxis={"gridcolor": "#333"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("Raw scalar data"):
                    st.dataframe(combined[combined["tag"].isin(selected_tags)], use_container_width=True)

        except ImportError:
            st.error("tbparse not installed. Run: pip install tbparse")
        except Exception as e:
            st.error(f"Error reading TensorBoard logs: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPERIMENTS.MD VIEWER
# ══════════════════════════════════════════════════════════════════════════════
with tab_exp:
    st.subheader("Experiments Log")

    if not EXPERIMENTS.exists():
        st.warning(f"`{EXPERIMENTS}` not found.")
    else:
        mtime = EXPERIMENTS.stat().st_mtime
        st.caption(f"Last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
        content = EXPERIMENTS.read_text()
        st.markdown(content, unsafe_allow_html=False)
