"""
Plotly figure builders for the ExoScope dashboard.

Every chart is constructed here so styling stays consistent and the page
modules read as layout rather than plotting code.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.theme import COLORS, SEQUENTIAL_SCALE, score_band, style_axes, PLOTLY_LAYOUT

EARTH_EQT = 255.0  # K — Earth's equilibrium temperature


def score_gauge(score: float, ci: tuple[float, float] | None = None,
                title: str = "HBLI Score") -> go.Figure:
    """Radial gauge for a 0-1 score, with the confidence interval marked."""
    _, color, _ = score_band(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 40, "family": "JetBrains Mono", "color": color}},
        title={"text": title, "font": {"size": 13, "color": COLORS["text_muted"]}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": COLORS["border_strong"],
                "tickfont": {"size": 10, "color": COLORS["text_faint"]},
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": COLORS["surface_alt"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(255,107,122,0.10)"},
                {"range": [25, 50], "color": "rgba(255,184,77,0.10)"},
                {"range": [50, 75], "color": "rgba(255,217,61,0.10)"},
                {"range": [75, 100], "color": "rgba(34,211,170,0.12)"},
            ],
            # The threshold marker shows the lower confidence bound, so the
            # gauge communicates the pessimistic case alongside the estimate.
            "threshold": {
                "line": {"color": COLORS["text"], "width": 2},
                "thickness": 0.8,
                "value": (ci[0] if ci else score) * 100,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=24, r=24, t=52, b=8), **{
        k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("margin",)
    })
    return fig


def subscore_bars(tabular: float, vision: float,
                  w_tab: float, w_vis: float) -> go.Figure:
    """Horizontal bars comparing the tabular and vision sub-scores."""
    labels, values, colors = [], [], []
    for name, value, weight, color in (
        ("Tabular habitability", tabular, w_tab, COLORS["primary"]),
        ("Vision biosignature", vision, w_vis, COLORS["accent"]),
    ):
        labels.append(f"{name}<br><span style='font-size:10px;color:{COLORS['text_faint']}'>"
                      f"weight {weight:.2f}</span>")
        values.append(value)
        colors.append(color if weight > 0 else COLORS["border_strong"])

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text_muted"], size=11),
        hovertemplate="%{x:.4f}<extra></extra>",
    ))
    style_axes(fig)
    fig.update_layout(height=250, showlegend=False, margin=dict(l=8, r=40, t=40, b=8),
                      title=dict(text="Sub-scores", font=dict(size=13, color=COLORS["text_muted"])))
    fig.update_xaxes(range=[0, 1.15])
    return fig


def habitable_zone_diagram(row: pd.Series) -> go.Figure | None:
    """Log-scale view of the star's habitable zone with the planet's orbit.

    Args:
        row: An engineered planet row carrying hz_dist_* columns.

    Returns:
        Figure, or None when the host star falls outside the Kopparapu fit.
    """
    needed = ["hz_dist_recent_venus", "hz_dist_runaway_greenhouse",
              "hz_dist_maximum_greenhouse", "hz_dist_early_mars"]
    if any(c not in row.index or pd.isna(row[c]) for c in needed):
        return None

    venus, runaway = float(row["hz_dist_recent_venus"]), float(row["hz_dist_runaway_greenhouse"])
    maxgreen, mars = float(row["hz_dist_maximum_greenhouse"]), float(row["hz_dist_early_mars"])
    a = float(row["pl_orbsmax"]) if "pl_orbsmax" in row.index and pd.notna(row["pl_orbsmax"]) else None

    fig = go.Figure()
    # Optimistic zone underneath, conservative zone on top of it.
    fig.add_shape(type="rect", x0=venus, x1=mars, y0=0.3, y1=0.7,
                  fillcolor="rgba(34,211,170,0.13)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=runaway, x1=maxgreen, y0=0.3, y1=0.7,
                  fillcolor="rgba(34,211,170,0.30)", line=dict(width=0), layer="below")

    # Invisible markers carry the hover text for each boundary.
    fig.add_trace(go.Scatter(
        x=[venus, runaway, maxgreen, mars], y=[0.5] * 4,
        mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"),
        hovertext=["Recent Venus (optimistic inner)", "Runaway greenhouse (conservative inner)",
                   "Maximum greenhouse (conservative outer)", "Early Mars (optimistic outer)"],
        hovertemplate="%{hovertext}<br>%{x:.3f} AU<extra></extra>",
        showlegend=False,
    ))

    if a is not None:
        in_cons = runaway >= a >= maxgreen
        in_opt = venus >= a >= mars
        color = COLORS["accent"] if in_cons else COLORS["warn"] if in_opt else COLORS["danger"]
        fig.add_trace(go.Scatter(
            x=[a], y=[0.5], mode="markers+text",
            marker=dict(size=17, color=color, symbol="circle",
                        line=dict(width=2, color=COLORS["bg"])),
            text=[f"  {row.get('pl_name', 'Planet')}"],
            textposition="top right",
            textfont=dict(color=color, size=12),
            hovertemplate=f"Orbit: {a:.4f} AU<extra></extra>",
            showlegend=False,
        ))

    # Earth reference, only when the scale makes it meaningful.
    if min(venus, a or venus) < 1.0 < max(mars, a or mars) * 4:
        fig.add_trace(go.Scatter(
            x=[1.0], y=[0.5], mode="markers+text",
            marker=dict(size=10, color=COLORS["info"], symbol="diamond",
                        line=dict(width=1.5, color=COLORS["bg"])),
            text=["Earth  "], textposition="bottom left",
            textfont=dict(color=COLORS["info"], size=10),
            hovertemplate="Earth: 1.000 AU<extra></extra>", showlegend=False,
        ))

    style_axes(fig, x_title="Orbital distance (AU, log scale)")
    fig.update_layout(
        height=210, showlegend=False,
        margin=dict(l=8, r=8, t=44, b=40),
        title=dict(text="Habitable zone — Kopparapu et al. (2014)",
                   font=dict(size=13, color=COLORS["text_muted"])),
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


def earth_comparison_radar(row: pd.Series) -> go.Figure | None:
    """Radar chart comparing a planet's key parameters to Earth.

    Each axis is a similarity score in [0, 1] — 1.0 means Earth-identical —
    so the axes are directly comparable rather than mixing raw units.
    """
    # (label, planet column, Earth value, scale for the difference)
    axes = [
        ("Radius", "pl_rade", 1.0, 1.0),
        ("Mass", "pl_bmasse", 1.0, 3.0),
        ("Density", "pl_dens", 5.51, 4.0),
        ("Insolation", "pl_insol", 1.0, 1.5),
        ("Temperature", "pl_eqt", EARTH_EQT, 120.0),
        ("Circular orbit", "pl_orbeccen", 0.0167, 0.35),
    ]

    labels, values = [], []
    for label, col, earth_value, scale in axes:
        if col not in row.index or pd.isna(row[col]):
            continue
        # Similarity falls off with the fractional difference from Earth.
        similarity = 1.0 / (1.0 + abs(float(row[col]) - earth_value) / scale)
        labels.append(label)
        values.append(round(similarity, 4))

    if len(labels) < 3:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * (len(labels) + 1), theta=labels + labels[:1],
        fill="toself", fillcolor="rgba(77,163,255,0.07)",
        line=dict(color=COLORS["info"], width=1.5, dash="dot"),
        name="Earth", hovertemplate="Earth: 1.000<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=values + values[:1], theta=labels + labels[:1],
        fill="toself", fillcolor="rgba(124,108,255,0.22)",
        line=dict(color=COLORS["primary"], width=2.5),
        name=str(row.get("pl_name", "Target")),
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=360, showlegend=True,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border"],
                            tickfont=dict(size=9, color=COLORS["text_faint"]),
                            angle=90, tickvals=[0.25, 0.5, 0.75, 1.0]),
            angularaxis=dict(gridcolor=COLORS["border"],
                             tickfont=dict(size=11, color=COLORS["text_muted"])),
        ),
    )
    fig.update_layout(title=dict(text="Similarity to Earth (1.0 = identical)",
                                 font=dict(size=13, color=COLORS["text_muted"])))
    return fig


def population_scatter(df: pd.DataFrame, x: str, y: str, color: str,
                       highlight: str | None = None, log_x: bool = True,
                       log_y: bool = False, labels: dict | None = None) -> go.Figure:
    """Population scatter of the catalog, optionally highlighting one planet."""
    labels = labels or {}
    plot_df = df.dropna(subset=[c for c in (x, y, color) if c in df.columns])

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=plot_df[x], y=plot_df[y],
        mode="markers",
        marker=dict(
            size=5,
            color=plot_df[color],
            colorscale=SEQUENTIAL_SCALE,
            showscale=True,
            opacity=0.72,
            line=dict(width=0),
            colorbar=dict(
                title=dict(text=labels.get(color, color), side="right",
                           font=dict(size=11, color=COLORS["text_muted"])),
                thickness=11, len=0.72, outlinewidth=0,
                tickfont=dict(size=10, color=COLORS["text_faint"]),
            ),
        ),
        customdata=plot_df[["pl_name"]] if "pl_name" in plot_df.columns else None,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" if "pl_name" in plot_df.columns else ""
        ) + f"{labels.get(x, x)}: %{{x:.3g}}<br>{labels.get(y, y)}: %{{y:.3g}}<extra></extra>",
        name="Confirmed planets",
    ))

    if highlight and "pl_name" in plot_df.columns:
        target = plot_df[plot_df["pl_name"] == highlight]
        if not target.empty:
            fig.add_trace(go.Scatter(
                x=target[x], y=target[y], mode="markers",
                marker=dict(size=15, color="rgba(0,0,0,0)", symbol="circle",
                            line=dict(width=2.5, color=COLORS["warn"])),
                name=highlight,
                hovertemplate=f"<b>{highlight}</b><extra></extra>",
            ))

    style_axes(fig, labels.get(x, x), labels.get(y, y))
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_layout(height=520, legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="right", x=1))
    return fig


def _name_fn(readable: Mapping[str, str] | Callable[[str], str]) -> Callable[[str], str]:
    """Normalise a label source into a lookup function.

    Accepts either a mapping or a naming callable, so callers can pass the
    dynamic resolver that also handles engineered suffixes and one-hot
    column names rather than only the static labels.
    """
    if callable(readable):
        return readable
    return lambda key: readable.get(key, key.replace("_", " "))


def importance_bars(importances: dict, readable, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of model feature importances."""
    label = _name_fn(readable)
    top = list(importances.items())[:top_n]
    names = [label(k) for k, _ in top][::-1]
    values = [v for _, v in top][::-1]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=values, colorscale=SEQUENTIAL_SCALE, line=dict(width=0)),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text_faint"], size=10),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    style_axes(fig, "Relative importance")
    fig.update_layout(
        height=max(320, 24 * len(top)), showlegend=False,
        margin=dict(l=8, r=56, t=16, b=32),
    )
    return fig


def shap_waterfall(contributions: list[tuple[str, float]], readable,
                   base_score: float) -> go.Figure:
    """Waterfall of the strongest SHAP contributions for one prediction."""
    label = _name_fn(readable)
    ranked = sorted(contributions, key=lambda kv: abs(kv[1]), reverse=True)[:10][::-1]
    names = [label(k) for k, _ in ranked]
    values = [v for _, v in ranked]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(
            color=[COLORS["accent"] if v > 0 else COLORS["danger"] for v in values],
            line=dict(width=0),
        ),
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text_faint"], size=10),
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=COLORS["border_strong"], width=1))
    style_axes(fig, "SHAP contribution (log-odds)")
    fig.update_layout(
        height=max(300, 26 * len(ranked)), showlegend=False,
        margin=dict(l=8, r=56, t=40, b=32),
        title=dict(text=f"Why this score? (model base rate {base_score:.3f})",
                   font=dict(size=13, color=COLORS["text_muted"])),
    )
    return fig


def discovery_timeline(df: pd.DataFrame) -> go.Figure:
    """Stacked discoveries per year, split by detection method."""
    if "disc_year" not in df.columns or "discoverymethod" not in df.columns:
        return go.Figure()

    counts = (
        df.dropna(subset=["disc_year"])
        .assign(disc_year=lambda d: d["disc_year"].astype(int))
        .groupby(["disc_year", "discoverymethod"]).size()
        .unstack(fill_value=0)
    )
    # Keep the methods that actually matter; fold the rest into "Other".
    top_methods = counts.sum().nlargest(4).index.tolist()
    if len(counts.columns) > len(top_methods):
        counts["Other"] = counts.drop(columns=top_methods).sum(axis=1)
        counts = counts[top_methods + ["Other"]]

    palette = [COLORS["primary"], COLORS["accent"], COLORS["info"],
               COLORS["warn"], COLORS["text_faint"]]

    fig = go.Figure()
    for i, method in enumerate(counts.columns):
        fig.add_trace(go.Bar(
            x=counts.index, y=counts[method], name=str(method),
            marker=dict(color=palette[i % len(palette)], line=dict(width=0)),
            hovertemplate=f"<b>{method}</b><br>%{{x}}: %{{y}} planets<extra></extra>",
        ))

    style_axes(fig, "Discovery year", "Planets discovered")
    fig.update_layout(
        barmode="stack", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=8, r=8, t=56, b=32),
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, columns: list[str],
                        readable) -> go.Figure:
    """Correlation matrix over the selected feature columns."""
    label = _name_fn(readable)
    available = [c for c in columns if c in df.columns]
    corr = df[available].corr()
    names = [label(c) for c in available]

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=names, y=names,
        colorscale=[[0, COLORS["danger"]], [0.5, COLORS["surface_alt"]], [1, COLORS["accent"]]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9, color=COLORS["text"]),
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(thickness=11, len=0.72, outlinewidth=0,
                      tickfont=dict(size=10, color=COLORS["text_faint"])),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=560)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=10, color=COLORS["text_muted"]))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10, color=COLORS["text_muted"]))
    return fig


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> go.Figure:
    """Reliability diagram comparing predicted probability to observed rate."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    centers, observed, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() > 0:
            centers.append(float(y_prob[mask].mean()))
            observed.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(color=COLORS["border_strong"], width=1.5, dash="dash"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=centers, y=observed, mode="markers+lines", name="Model",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=[min(6 + c / 40, 20) for c in counts], color=COLORS["primary"]),
        customdata=counts,
        hovertemplate="Predicted %{x:.3f}<br>Observed %{y:.3f}<br>n = %{customdata}<extra></extra>",
    ))
    style_axes(fig, "Mean predicted probability", "Observed frequency")
    fig.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="left", x=0))
    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])
    return fig


def detection_class_bars(class_counts: dict) -> go.Figure:
    """Bar chart of vision detections per biosignature proxy class."""
    names = [k.replace("_", " ").title() for k in class_counts]
    palette = [COLORS["accent"], COLORS["warn"], COLORS["primary"], COLORS["info"]]

    fig = go.Figure(go.Bar(
        x=names, y=list(class_counts.values()),
        marker=dict(color=palette[:len(class_counts)], line=dict(width=0)),
        text=list(class_counts.values()),
        textposition="outside",
        textfont=dict(color=COLORS["text_muted"], size=11),
        hovertemplate="%{x}: %{y} detections<extra></extra>",
    ))
    style_axes(fig, "", "Detections")
    fig.update_layout(height=320, showlegend=False, margin=dict(l=8, r=8, t=48, b=32),
                      title=dict(text="Detections by proxy class",
                                 font=dict(size=13, color=COLORS["text_muted"])))
    return fig
