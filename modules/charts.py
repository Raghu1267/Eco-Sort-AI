"""Plotly charts built only from the current session's real detection results."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.segregation import HAZARD_COLORS, HAZARD_ORDER, count_by

_TEAL = "#2dd4bf"
_FONT = dict(family="Inter, sans-serif", color="#e6edf7")


def _style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_FONT,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(font=dict(size=15)),
    )
    fig.update_xaxes(gridcolor="#1f3050", zerolinecolor="#1f3050")
    fig.update_yaxes(gridcolor="#1f3050", zerolinecolor="#1f3050")
    return fig


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data", showarrow=False, font=dict(size=14, color="#8fa3bf"))
    fig.update_layout(title=title)
    return _style(fig)


def bar_counts(df: pd.DataFrame, column: str, title: str, horizontal: bool = True) -> go.Figure:
    counts = count_by(df, column)
    if counts.empty:
        return _empty(title)
    counts = counts.sort_values("Count", ascending=True)
    if horizontal:
        fig = px.bar(counts, x="Count", y=column, orientation="h", title=title, text="Count")
    else:
        fig = px.bar(counts, x=column, y="Count", title=title, text="Count")
    fig.update_traces(marker_color=_TEAL, textposition="outside", cliponaxis=False)
    if horizontal:
        fig.update_yaxes(title=None)
        fig.update_xaxes(dtick=1, title="Objects")
        return _style(fig, height=max(300, 60 + 38 * len(counts)))
    fig.update_xaxes(title=None)
    fig.update_yaxes(dtick=1, title="Objects")
    return _style(fig, height=340)


def confidence_histogram(df: pd.DataFrame) -> go.Figure:
    title = "Confidence distribution"
    if df.empty:
        return _empty(title)
    fig = px.histogram(df, x="Confidence (%)", nbins=10, range_x=[0, 100], title=title)
    fig.update_traces(marker_color="#3ddc97", marker_line_color="#0b1220", marker_line_width=1)
    fig.update_layout(yaxis_title="Objects", xaxis_title="Confidence (%)", bargap=0.05)
    return _style(fig)


def hazard_donut(df: pd.DataFrame) -> go.Figure:
    title = "Hazard-priority distribution"
    counts = count_by(df, "Hazard Priority", HAZARD_ORDER)
    if counts.empty:
        return _empty(title)
    fig = px.pie(
        counts, names="Hazard Priority", values="Count", hole=0.55, title=title,
        color="Hazard Priority", color_discrete_map=HAZARD_COLORS,
        category_orders={"Hazard Priority": HAZARD_ORDER},
    )
    fig.update_traces(textinfo="label+value", sort=False)
    return _style(fig)
