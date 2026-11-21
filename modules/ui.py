"""Styling and small HTML components for the ECO SORT AI dashboard."""
from __future__ import annotations

from html import escape
from typing import Iterable

import streamlit as st

from modules.segregation import HAZARD_COLORS

NAVY = "#0b1220"
PANEL = "#111b2e"
PANEL_2 = "#16233b"
BORDER = "#1f3050"
TEXT = "#e6edf7"
MUTED = "#8fa3bf"
GREEN = "#3ddc97"
TEAL = "#2dd4bf"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700;800&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.stApp {{
  background:
    radial-gradient(1100px 500px at 85% -10%, rgba(45,212,191,.10), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(61,220,151,.07), transparent 55%),
    {NAVY};
  color: {TEXT};
}}
h1, h2, h3, h4 {{ font-family: 'Sora', 'Inter', sans-serif; letter-spacing: -0.01em; color: {TEXT}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0d1628 0%, #0a1322 100%);
  border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {{
  border-radius: 10px; margin: 2px 6px; padding: .45rem .7rem;
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: rgba(45,212,191,.14); border: 1px solid rgba(45,212,191,.35);
}}

/* Hero */
.es-hero {{
  background: linear-gradient(135deg, rgba(61,220,151,.16), rgba(45,212,191,.06) 55%, rgba(17,27,46,.9));
  border: 1px solid {BORDER}; border-radius: 22px; padding: 2.4rem 2.2rem; margin-bottom: 1.4rem;
}}
.es-hero h1 {{ font-size: 3rem; font-weight: 800; margin: 0 0 .2rem 0; line-height: 1.05; }}
.es-hero h1 span {{ background: linear-gradient(90deg, {GREEN}, {TEAL}); -webkit-background-clip: text;
  background-clip: text; color: transparent; }}
.es-hero p.sub {{ color: {TEAL}; font-weight: 600; font-size: 1.05rem; margin: 0 0 .9rem 0; }}
.es-hero p.desc {{ color: {MUTED}; max-width: 62ch; line-height: 1.65; margin: 0; }}

/* Page header */
.es-page-title {{ font-family: 'Sora', sans-serif; font-size: 1.9rem; font-weight: 700; margin: 0; }}
.es-page-sub {{ color: {MUTED}; margin: .25rem 0 1.3rem 0; }}

/* Cards */
.es-card {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.15rem 1.25rem;
  height: 100%;
}}
.es-card h4 {{ margin: 0 0 .35rem 0; font-size: 1.02rem; }}
.es-card p {{ margin: 0; color: {MUTED}; line-height: 1.6; font-size: .94rem; }}
.es-card .ico {{ font-size: 1.5rem; margin-bottom: .45rem; display: block; }}
.es-stat {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1rem 1.2rem; }}
.es-stat .v {{ font-family: 'Sora', sans-serif; font-size: 1.9rem; font-weight: 700; color: {GREEN}; line-height: 1.15; }}
.es-stat .l {{ color: {MUTED}; font-size: .88rem; margin-top: .15rem; }}
.es-step {{ border-left: 3px solid {TEAL}; background: {PANEL}; border-radius: 4px 14px 14px 4px;
  padding: .85rem 1.1rem; margin-bottom: .6rem; }}
.es-step b {{ color: {TEXT}; }} .es-step span {{ color: {MUTED}; }}

/* Badges */
.es-badge {{ display: inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .8rem;
  font-weight: 600; border: 1px solid; }}
.es-field {{ margin-bottom: .75rem; }}
.es-field .k {{ color: {MUTED}; font-size: .8rem; margin-bottom: .1rem; }}
.es-field .t {{ color: {TEXT}; line-height: 1.55; font-size: .95rem; }}
.es-warn {{ border-left: 3px solid #ff5d6c; background: rgba(255,93,108,.08); border-radius: 4px 12px 12px 4px;
  padding: .75rem 1rem; }}
.es-note {{ border-left: 3px solid {TEAL}; background: rgba(45,212,191,.07); border-radius: 4px 12px 12px 4px;
  padding: .75rem 1rem; color: {MUTED}; font-size: .92rem; line-height: 1.6; }}
.es-bin {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: .9rem 1rem; }}
.es-bin .n {{ font-family: 'Sora', sans-serif; font-size: 1.6rem; font-weight: 700; color: {TEAL}; }}
.es-bin .b {{ color: {MUTED}; font-size: .86rem; }}

/* Streamlit widgets */
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 16px; padding: .9rem 1.1rem; }}
div[data-testid="stMetricValue"] {{ color: {GREEN}; font-family: 'Sora', sans-serif; }}
div[data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background: linear-gradient(90deg, {GREEN}, {TEAL}); color: #06251c; border: 0; border-radius: 12px; font-weight: 600;
}}
.stButton > button, .stDownloadButton > button {{ border-radius: 12px; }}
div[data-testid="stFileUploader"] section {{ background: {PANEL}; border: 1.5px dashed #2c4a72; border-radius: 16px; }}
div[data-testid="stExpander"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
hr {{ border-color: {BORDER}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _html(markup: str) -> None:
    # Strip indentation/newlines so Markdown never treats the HTML as a code block.
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    _html(f'<div class="es-page-title">{escape(title)}</div><div class="es-page-sub">{escape(subtitle)}</div>')


def hero() -> None:
    _html(
        """
        <div class="es-hero">
          <h1><span>ECO SORT AI</span></h1>
          <p class="sub">AI-Based Smart E-Waste Segregation and Classification System</p>
          <p class="desc">Upload a photo of discarded electronics. A YOLOv8s model detects each item, and a
          rule-based engine tells you which collection stream it belongs to, how hazardous it is, and how to
          handle it safely.</p>
        </div>
        """
    )


def stat_card(value: str, label: str) -> None:
    _html(f'<div class="es-stat"><div class="v">{escape(str(value))}</div><div class="l">{escape(label)}</div></div>')


def info_card(icon: str, title: str, body: str) -> None:
    _html(
        f'<div class="es-card"><span class="ico">{icon}</span><h4>{escape(title)}</h4><p>{escape(body)}</p></div>'
    )


def step_card(title: str, body: str) -> None:
    _html(f'<div class="es-step"><b>{escape(title)}</b><br><span>{escape(body)}</span></div>')


def hazard_badge(hazard: str) -> str:
    color = HAZARD_COLORS.get(hazard, "#8fa3bf")
    return (
        f'<span class="es-badge" style="color:{color};border-color:{color};background:{color}1f">'
        f"{escape(hazard)} hazard</span>"
    )


def field(label: str, text_html: str) -> str:
    return f'<div class="es-field"><div class="k">{escape(label)}</div><div class="t">{text_html}</div></div>'


def render_html(markup: str) -> None:
    _html(markup)


def note(text: str) -> None:
    _html(f'<div class="es-note">{escape(text)}</div>')


def warning_box(title: str, items: Iterable[str]) -> None:
    lis = "".join(f"<li>{escape(i)}</li>" for i in items)
    _html(f'<div class="es-warn"><b>{escape(title)}</b><ul style="margin:.4rem 0 0 1.1rem">{lis}</ul></div>')


def bin_card(count: int, name: str) -> None:
    _html(f'<div class="es-bin"><div class="n">{count}</div><div class="b">{escape(name)}</div></div>')
