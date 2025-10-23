from __future__ import annotations

import os
from functools import lru_cache
from typing import List, TypedDict

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ANALYTICS_BASE = os.getenv("ANALYTICS_API_URL", "http://localhost:8080")
COACH_BASE = os.getenv("COACH_API_URL", "http://localhost:8082")

st.set_page_config(page_title="ACC Coach Dashboard", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --color-bg: #050608;
        --color-surface: rgba(12, 13, 16, 0.92);
        --color-border: rgba(249, 200, 70, 0.2);
        --color-primary: #d63224;
        --color-highlight: #f9c846;
        --color-text: #f5f5f5;
        --radius-xl: 28px;
        --radius-lg: 20px;
        --shadow-strong: 0 24px 48px rgba(214, 50, 36, 0.24);
    }
    .acc-card {
        background: linear-gradient(150deg, rgba(214,50,36,0.22), rgba(8,9,12,0.78));
        border-radius: var(--radius-xl);
        border: 1px solid var(--color-border);
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: var(--shadow-strong);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 18px;
    }
    .metric-pill {
        background: rgba(8, 9, 12, 0.85);
        border-radius: 22px;
        padding: 16px 18px;
        border: 1px solid rgba(249,200,70,0.16);
        color: var(--color-text);
        text-align: center;
    }
    .metric-pill .label {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.7;
    }
    .metric-pill .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--color-highlight);
    }
    .coach-box {
        border-radius: var(--radius-lg);
        border: 1px dashed rgba(249,200,70,0.35);
        padding: 18px 22px;
        background: rgba(214, 50, 36, 0.12);
    }
    .coach-box h4 {
        color: var(--color-highlight);
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

class SectionIssuePayload(TypedDict):
    section: str
    delta_time_ms: float
    cause: str
    suggestion: str


@lru_cache
def _get_sessions() -> List[dict]:
    response = requests.get(f"{ANALYTICS_BASE}/sessions", timeout=5)
    response.raise_for_status()
    return response.json()


@lru_cache
def _get_session_report(session_id: str) -> dict:
    response = requests.get(f"{ANALYTICS_BASE}/sessions/{session_id}", timeout=5)
    response.raise_for_status()
    return response.json()


def fetch_coach_feedback(
    session_id: str, lap: int, issues: List[SectionIssuePayload]
) -> str:
    payload = {
        "session_id": session_id,
        "lap": lap,
        "language": "it",
        "tone": "coach",
        "driver_level": "intermediate",
        "summary": "Genera suggerimenti sintetici basati sulle sezioni critiche",
        "issues": issues,
        "metrics": {},
    }
    response = requests.post(
        f"{COACH_BASE}/coach/realtime",
        json=payload,
        timeout=8,
    )
    response.raise_for_status()
    return response.json()["message"]


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-pill">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sections_table(sections: List[dict]) -> None:
    if not sections:
        st.info("Nessuna sezione critica registrata.")
        return
    df = pd.DataFrame(sections)
    df["delta_time_ms"] = df["delta_time_ms"].round(1)
    df.rename(
        columns={
            "section_name": "Sezione",
            "delta_time_ms": "Delta ms",
            "avg_speed_kph": "Velocita media",
            "throttle_avg": "Throttle",
            "brake_avg": "Brake",
        },
        inplace=True,
    )
    st.dataframe(
        df[["Sezione", "Delta ms", "Velocita media", "Throttle", "Brake"]],
        use_container_width=True,
        hide_index=True,
    )


def render_section_chart(sections: List[dict]) -> None:
    if not sections:
        return
    labels = [s["section_name"] for s in sections]
    values = [s["delta_time_ms"] for s in sections]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color="#d63224", line=dict(color="#f9c846", width=1.5)),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="Delta per sezione (ms)",
        paper_bgcolor="rgba(8,9,12,0.85)",
        plot_bgcolor="rgba(8,9,12,0.3)",
        margin=dict(t=60, r=24, b=40, l=24),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_lap_chart(laps: List[dict]) -> None:
    if not laps:
        return
    df = pd.DataFrame(laps)
    df["lap_time_sec"] = df["lap_time_ms"] / 1000
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["lap"],
                y=df["lap_time_sec"],
                mode="lines+markers",
                line=dict(color="#f9c846", shape="spline"),
                marker=dict(size=10, color="#d63224"),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="Tempi sul giro",
        xaxis_title="Lap",
        yaxis_title="Secondi",
        paper_bgcolor="rgba(8,9,12,0.85)",
        plot_bgcolor="rgba(8,9,12,0.3)",
        margin=dict(t=60, r=24, b=40, l=24),
    )
    st.plotly_chart(fig, use_container_width=True)


st.sidebar.title("ACC Coach Dashboard")
sessions = _get_sessions()
session_map = {f"{s['track_name']} - {s['started_at']}": s["id"] for s in sessions}

session_choice = st.sidebar.selectbox(
    "Seleziona sessione",
    options=list(session_map.keys()),
)

selected_session_id = session_map.get(session_choice)

if not selected_session_id:
    st.warning("Nessuna sessione disponibile.")
    st.stop()

report = _get_session_report(selected_session_id)

st.title("Panoramica sessione")
st.subheader(f"{report['session']['track_name']} - {report['session']['car_model']}")

cols = st.columns(4)
with cols[0]:
    render_metric("Lap totali", str(len(report["laps"])))
with cols[1]:
    best = min((lap["lap_time_ms"] for lap in report["laps"] if lap["lap_time_ms"]), default=0)
    render_metric("Best lap", f"{best/1000:.3f} s")
with cols[2]:
    render_metric("Consistenza", f"{(report['session']['consistency_score'] or 0.0)*100:.0f}%")
with cols[3]:
    render_metric("Efficienza", f"{(report['session']['efficiency_score'] or 0.0)*100:.0f}%")

st.markdown('<div class="acc-card">', unsafe_allow_html=True)
st.markdown("### Delta e sezioni critiche")
render_section_chart(report["critical_sections"])
render_sections_table(report["critical_sections"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="acc-card">', unsafe_allow_html=True)
st.markdown("### Trend tempi sul giro")
render_lap_chart(report["laps"])
st.markdown("</div>", unsafe_allow_html=True)

if st.sidebar.button("Genera coaching mirato"):
    try:
        issues: List[SectionIssuePayload] = [
            {
                "section": section["section_name"],
                "delta_time_ms": section["delta_time_ms"],
                "cause": "tempo perso rispetto best",
                "suggestion": "Sposta il punto di frenata piu avanti e dosa il rilascio",
            }
            for section in report["critical_sections"]
        ]
        message = fetch_coach_feedback(selected_session_id, report["laps"][0]["lap"], issues)
        st.sidebar.success("Feedback generato")
        st.markdown('<div class="coach-box">', unsafe_allow_html=True)
        st.markdown("#### Coach AI")
        st.write(message)
        st.markdown("</div>", unsafe_allow_html=True)
    except requests.RequestException as exc:
        st.sidebar.error(f"Errore coach AI: {exc}")
