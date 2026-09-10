"""
Design tokens and shared UI primitives for the ExoScope dashboard.

Everything visual lives here so the pages stay readable: one palette, one
type scale, one set of card/badge/metric builders. Colours are defined as
CSS custom properties and mirrored in Python for Plotly, which cannot read
CSS variables.
"""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st

# ── Palette ────────────────────────────────────────────────────────────
# Mirrored in .streamlit/config.toml. Chosen for legibility on a dark
# ground: the sequence below is distinguishable in both common forms of
# red-green colour blindness, and every text pairing clears WCAG AA.
COLORS = {
    "bg": "#0A0C12",
    "surface": "#12151F",
    "surface_alt": "#171B28",
    "surface_hover": "#1D2231",
    "border": "#252B3B",
    "border_strong": "#333A4F",
    "text": "#ECEEF4",
    "text_muted": "#98A0B5",
    "text_faint": "#6B7386",
    "primary": "#7C6CFF",
    "primary_soft": "#A79BFF",
    "accent": "#22D3AA",
    "warn": "#FFB84D",
    "danger": "#FF6B7A",
    "info": "#4DA3FF",
}

# Score bands, shared by gauges, badges and tables.
SCORE_BANDS = [
    (0.75, "High Likelihood", COLORS["accent"], "high"),
    (0.50, "Moderate Likelihood", "#FFD93D", "moderate"),
    (0.25, "Low Likelihood", COLORS["warn"], "low"),
    (0.00, "Very Low Likelihood", COLORS["danger"], "verylow"),
]

# Sequential scale for continuous quantities (ESI, scores).
SEQUENTIAL_SCALE = [
    [0.00, "#2A2F45"],
    [0.25, "#4A5BC4"],
    [0.50, "#7C6CFF"],
    [0.75, "#3FBFA8"],
    [1.00, "#22D3AA"],
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=16, r=16, t=48, b=16),
    hoverlabel=dict(
        bgcolor=COLORS["surface_alt"],
        bordercolor=COLORS["border_strong"],
        font=dict(color=COLORS["text"], size=12),
    ),
    legend=dict(font=dict(color=COLORS["text_muted"], size=11)),
)

AXIS_STYLE = dict(
    gridcolor=COLORS["border"],
    zerolinecolor=COLORS["border_strong"],
    color=COLORS["text_muted"],
    title_font=dict(size=12, color=COLORS["text_muted"]),
    tickfont=dict(size=11),
)


def score_band(score: float) -> tuple[str, str, str]:
    """Return (label, hex colour, css slug) for a 0-1 score."""
    for threshold, label, color, slug in SCORE_BANDS:
        if score >= threshold:
            return label, color, slug
    return SCORE_BANDS[-1][1], SCORE_BANDS[-1][2], SCORE_BANDS[-1][3]


def style_axes(fig, x_title: str = "", y_title: str = ""):
    """Apply the shared layout and axis styling to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(**AXIS_STYLE, title_text=x_title or None)
    fig.update_yaxes(**AXIS_STYLE, title_text=y_title or None)
    return fig


# ── Global stylesheet ──────────────────────────────────────────────────
def inject_css() -> None:
    """Inject the dashboard stylesheet. Call once, early, per page load."""
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --bg: {COLORS['bg']};
  --surface: {COLORS['surface']};
  --surface-alt: {COLORS['surface_alt']};
  --surface-hover: {COLORS['surface_hover']};
  --border: {COLORS['border']};
  --border-strong: {COLORS['border_strong']};
  --text: {COLORS['text']};
  --muted: {COLORS['text_muted']};
  --faint: {COLORS['text_faint']};
  --primary: {COLORS['primary']};
  --primary-soft: {COLORS['primary_soft']};
  --accent: {COLORS['accent']};
  --warn: {COLORS['warn']};
  --danger: {COLORS['danger']};
  --info: {COLORS['info']};
  --radius: 14px;
  --radius-sm: 10px;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}}

html, body, [class*="st-"], button, input, textarea {{
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}

/* Streamlit draws its chevrons and icons as Material Symbols ligatures.
   The blanket font rule above matches their emotion-cache classes, which
   turns each icon into its literal name ("keyboard_arrow_right"), so the
   icon font has to be restored explicitly. */
[data-testid="stIconMaterial"], span[data-testid^="stIconMaterial"] {{
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
}}

.stApp {{ background: var(--bg); }}

/* Give the content column room to breathe without going edge-to-edge. */
.block-container {{
  padding-top: 2.2rem;
  padding-bottom: 4rem;
  max-width: 1500px;
}}

/* ── Masthead ─────────────────────────────────────────────────────── */
.masthead {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 2rem;
  padding: 1.6rem 1.9rem;
  margin-bottom: 1.25rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background:
    radial-gradient(120% 180% at 0% 0%, rgba(124,108,255,0.16), transparent 55%),
    radial-gradient(120% 180% at 100% 0%, rgba(34,211,170,0.10), transparent 55%),
    var(--surface);
}}
.masthead h1 {{
  margin: 0;
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: -0.025em;
  color: var(--text);
}}
.masthead h1 .accent {{ color: var(--primary-soft); }}
.masthead p {{
  margin: 0.45rem 0 0;
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.55;
  max-width: 68ch;
}}
.masthead-meta {{
  text-align: right;
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--faint);
  white-space: nowrap;
  line-height: 1.7;
}}
.masthead-meta b {{ color: var(--muted); font-weight: 500; }}

/* ── Provenance banner ────────────────────────────────────────────── */
.provenance {{
  display: flex;
  align-items: center;
  gap: 0.85rem;
  padding: 0.7rem 1.05rem;
  margin-bottom: 1.5rem;
  border-radius: var(--radius-sm);
  font-size: 0.85rem;
  line-height: 1.5;
  border: 1px solid transparent;
}}
.provenance .dot {{
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}}
.provenance.live {{
  background: rgba(34,211,170,0.07);
  border-color: rgba(34,211,170,0.28);
  color: #9BE9D4;
}}
.provenance.live .dot {{ background: var(--accent); box-shadow: 0 0 0 4px rgba(34,211,170,0.16); }}
.provenance.synthetic {{
  background: rgba(255,107,122,0.09);
  border-color: rgba(255,107,122,0.35);
  color: #FFB3BB;
}}
.provenance.synthetic .dot {{ background: var(--danger); box-shadow: 0 0 0 4px rgba(255,107,122,0.16); }}
.provenance.stale {{
  background: rgba(255,184,77,0.08);
  border-color: rgba(255,184,77,0.3);
  color: #FFD9A0;
}}
.provenance.stale .dot {{ background: var(--warn); box-shadow: 0 0 0 4px rgba(255,184,77,0.16); }}
.provenance code {{
  font-family: var(--mono);
  background: rgba(255,255,255,0.06);
  padding: 0.08rem 0.4rem;
  border-radius: 5px;
  font-size: 0.78rem;
}}

/* ── Metric cards ─────────────────────────────────────────────────── */
.metric-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.8rem;
  margin-bottom: 0.4rem;
}}
.metric {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.95rem 1.1rem;
  transition: border-color 0.18s ease, transform 0.18s ease;
}}
.metric:hover {{ border-color: var(--border-strong); transform: translateY(-1px); }}
.metric .value {{
  font-family: var(--mono);
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1.15;
  letter-spacing: -0.02em;
}}
.metric .value.na {{ color: var(--faint); font-size: 1.2rem; }}
/* Word-shaped values ("Conservative HZ") need a smaller size than the
   numeric ones, or they wrap mid-word inside the card. */
.metric .value.text {{
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 0.98rem;
  font-weight: 600;
  line-height: 1.35;
  letter-spacing: 0;
  padding: 0.22rem 0;
}}
.metric .label {{
  color: var(--muted);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-top: 0.4rem;
  font-weight: 500;
}}
.metric .sub {{
  color: var(--faint);
  font-size: 0.72rem;
  margin-top: 0.25rem;
  font-family: var(--mono);
}}
.metric.accent .value {{ color: var(--accent); }}
.metric.primary .value {{ color: var(--primary-soft); }}

/* ── Section headers ──────────────────────────────────────────────── */
.section {{
  display: flex;
  align-items: baseline;
  gap: 0.7rem;
  margin: 2rem 0 0.9rem;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--border);
}}
.section h2 {{
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.01em;
}}
.section span {{ color: var(--faint); font-size: 0.82rem; }}

/* ── Badges ───────────────────────────────────────────────────────── */
.badge {{
  display: inline-block;
  padding: 0.28rem 0.8rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  border: 1px solid;
}}
.badge-high {{ background: rgba(34,211,170,0.13); color: #22D3AA; border-color: rgba(34,211,170,0.35); }}
.badge-moderate {{ background: rgba(255,217,61,0.13); color: #FFD93D; border-color: rgba(255,217,61,0.35); }}
.badge-low {{ background: rgba(255,184,77,0.13); color: #FFB84D; border-color: rgba(255,184,77,0.35); }}
.badge-verylow {{ background: rgba(255,107,122,0.13); color: #FF6B7A; border-color: rgba(255,107,122,0.35); }}
.badge-neutral {{ background: rgba(152,160,181,0.12); color: var(--muted); border-color: var(--border-strong); }}

/* ── Panels ───────────────────────────────────────────────────────── */
.panel {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.4rem;
  height: 100%;
}}
.panel h3, .panel-title {{
  margin: 0 0 0.9rem;
  font-size: 0.74rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}}

/* Panels that hold live widgets.

   Streamlit sanitizes each markdown block on its own, so an opening <div>
   in one st.markdown call is auto-closed before the next widget renders —
   the "open a div, draw a widget, close the div" pattern silently
   produces an empty box with the content stranded underneath. A real
   st.container is the only way to wrap widgets.

   The selector keys off the .panel-title marker this file emits rather
   than a Streamlit testid, because those are renamed between versions
   (stVerticalBlockBorderWrapper became stLayoutWrapper in 1.63). The
   :not(...) clause keeps only the innermost matching wrapper, so nested
   containers do not draw a double border. */
[data-testid="stLayoutWrapper"]:has(.panel-title):not(:has([data-testid="stLayoutWrapper"] .panel-title)),
[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-title):not(:has([data-testid="stVerticalBlockBorderWrapper"] .panel-title)) {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.15rem 1.3rem;
}}
.panel-title {{ margin-bottom: 0.7rem; }}

/* Attribution rows */
.factor {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.42rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.035);
  font-size: 0.85rem;
}}
.factor:last-child {{ border-bottom: none; }}
.factor .name {{ flex: 1; color: var(--text); }}
.factor .val {{ font-family: var(--mono); font-size: 0.8rem; font-weight: 500; }}
.factor .bar {{
  height: 4px; border-radius: 2px; flex-shrink: 0; min-width: 3px; opacity: 0.75;
}}
.factor.pos .val, .factor.pos .name strong {{ color: var(--accent); }}
.factor.pos .bar {{ background: var(--accent); }}
.factor.neg .val {{ color: var(--danger); }}
.factor.neg .bar {{ background: var(--danger); }}

/* Key-value list for planet facts */
.kv {{ display: grid; grid-template-columns: auto 1fr; gap: 0.35rem 1rem; font-size: 0.85rem; }}
.kv dt {{ color: var(--muted); white-space: nowrap; }}
.kv dd {{ margin: 0; color: var(--text); font-family: var(--mono); font-size: 0.82rem; text-align: right; }}
.kv dd.na {{ color: var(--faint); }}

/* ── Disclaimer ───────────────────────────────────────────────────── */
.disclaimer {{
  background: rgba(255,184,77,0.06);
  border: 1px solid rgba(255,184,77,0.22);
  border-left: 3px solid var(--warn);
  border-radius: var(--radius-sm);
  padding: 0.85rem 1.1rem;
  font-size: 0.8rem;
  line-height: 1.6;
  color: #E8D5B0;
}}
.disclaimer b {{ color: #FFD9A0; }}

.note {{
  color: var(--faint);
  font-size: 0.78rem;
  line-height: 1.6;
  font-style: normal;
}}

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
  background: #0D101A;
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
.sidebar-brand {{
  display: flex; align-items: center; gap: 0.65rem;
  padding-bottom: 0.9rem; margin-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
}}
.sidebar-brand .glyph {{ font-size: 1.5rem; line-height: 1; }}
.sidebar-brand .name {{ font-size: 1.12rem; font-weight: 700; color: var(--text); letter-spacing: -0.01em; }}
.sidebar-brand .tag {{ font-size: 0.68rem; color: var(--faint); text-transform: uppercase; letter-spacing: 0.09em; }}
.sidebar-label {{
  font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.09em; color: var(--faint); margin: 1.1rem 0 0.4rem;
}}

/* ── Streamlit widget polish ──────────────────────────────────────── */
.stButton > button {{
  border-radius: var(--radius-sm);
  font-weight: 600;
  font-size: 0.87rem;
  border: 1px solid var(--border-strong);
  transition: all 0.16s ease;
}}
.stButton > button[kind="primary"] {{
  background: var(--primary);
  border-color: var(--primary);
}}
.stButton > button[kind="primary"]:hover {{
  background: var(--primary-soft);
  border-color: var(--primary-soft);
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.3rem;
  border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
  font-weight: 500;
  font-size: 0.88rem;
  padding: 0.55rem 1rem;
  color: var(--muted);
}}
.stTabs [aria-selected="true"] {{ color: var(--text); }}

[data-testid="stMetricValue"] {{ font-family: var(--mono); }}

div[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}}

#MainMenu, footer {{ visibility: hidden; }}

.appfooter {{
  margin-top: 3rem; padding-top: 1.1rem;
  border-top: 1px solid var(--border);
  text-align: center; color: var(--faint); font-size: 0.75rem; line-height: 1.7;
}}
</style>
""",
        unsafe_allow_html=True,
    )


# ── Component builders ─────────────────────────────────────────────────
def section(title: str, hint: str = "") -> None:
    """Render a section header with an optional hint."""
    hint_html = f"<span>{hint}</span>" if hint else ""
    st.markdown(f'<div class="section"><h2>{title}</h2>{hint_html}</div>',
                unsafe_allow_html=True)


def metric_row(metrics: list[tuple]) -> None:
    """Render a responsive row of metric cards.

    Args:
        metrics: (label, value) or (label, value, sub) or
            (label, value, sub, variant) tuples, where variant is
            "", "accent" or "primary".
    """
    cards = []
    for item in metrics:
        label, value = item[0], item[1]
        sub = item[2] if len(item) > 2 else ""
        variant = item[3] if len(item) > 3 else ""

        is_na = value in (None, "", "N/A") or (isinstance(value, float) and value != value)
        # Numeric-looking values get the mono treatment; word values do not.
        is_numeric = isinstance(value, (int, float)) or (
            isinstance(value, str)
            and value.replace(",", "").replace(".", "").replace("%", "")
                     .replace("-", "").replace("+", "").isdigit()
        )
        value_cls = "value na" if is_na else "value" if is_numeric else "value text"
        display = "—" if is_na else value
        sub_html = f'<div class="sub">{sub}</div>' if sub else ""
        cards.append(
            f'<div class="metric {variant}"><div class="{value_cls}">{display}</div>'
            f'<div class="label">{label}</div>{sub_html}</div>'
        )
    st.markdown(f'<div class="metric-row">{"".join(cards)}</div>', unsafe_allow_html=True)


@contextmanager
def panel(title: str = ""):
    """A bordered panel that can actually contain Streamlit widgets.

    Use this instead of hand-writing ``<div class="panel">`` whenever the
    panel needs to hold a chart, dataframe or input — raw HTML div tags
    cannot wrap widgets across separate ``st.markdown`` calls.

    Args:
        title: Optional uppercase panel heading.

    Yields:
        The Streamlit container, so callers can also use it directly.
    """
    container = st.container(border=True)
    with container:
        if title:
            st.markdown(f'<div class="panel-title">{title}</div>',
                        unsafe_allow_html=True)
        yield container


def badge(text: str, variant: str = "neutral") -> str:
    """Return badge HTML. Variants: high, moderate, low, verylow, neutral."""
    return f'<span class="badge badge-{variant}">{text}</span>'


def disclaimer(html: str) -> None:
    """Render the standard warning-styled disclaimer block."""
    st.markdown(f'<div class="disclaimer">{html}</div>', unsafe_allow_html=True)
