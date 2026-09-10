"""
ExoScope — Streamlit Web Dashboard

The primary UI for the Habitability & Biosignature-Likelihood Index.
Provides interactive exploration of the NASA Exoplanet Archive, surface
image analysis, and fused HBLI scoring with explainable breakdowns.

Layout lives here; styling is in app/theme.py, figures in app/charts.py,
and data loading in app/state.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="ExoScope — Habitability Index",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app import charts  # noqa: E402
from app.state import (  # noqa: E402
    READABLE, ensure_data, feature_columns, init_state, provenance_banner, readable_name,
)
from app.theme import (  # noqa: E402
    COLORS, badge, disclaimer, inject_css, metric_row, panel, score_band, section,
)
from app.views import image_analysis, model_lab  # noqa: E402
from src.models.fusion import FusionLayer  # noqa: E402
from src.models.tabular_model import xgboost_available  # noqa: E402

inject_css()
init_state()

MODES = {
    "Target Analysis": "Score a single planet and see what drove the result",
    "Catalog Explorer": "Explore all confirmed planets across the archive",
    "Model Lab": "Train, evaluate and inspect the habitability model",
    "Surface Imagery": "Run the vision pipeline over planetary imagery",
}


# ── Sidebar ────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    """Render the sidebar and return the selected mode."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><span class="glyph">🔭</span><div>'
            '<div class="name">ExoScope</div>'
            '<div class="tag">Habitability Index</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Workspace", list(MODES), label_visibility="collapsed",
            captions=list(MODES.values()),
        )

        st.markdown('<div class="sidebar-label">Data source</div>', unsafe_allow_html=True)
        meta = st.session_state.get("meta")
        if meta is not None:
            tone = "🔴" if meta.is_synthetic else "🟢"
            st.caption(
                f"{tone} **{meta.n_rows:,}** planets · {meta.n_columns} columns  \n"
                f"`{meta.table}` · {meta.fetched_at_display}"
            )
        else:
            st.caption("Not loaded")

        if st.button("↻ Refresh from NASA", use_container_width=True,
                     help="Re-query the Exoplanet Archive and rebuild all features"):
            st.cache_data.clear()
            ensure_data(force_refresh=True)
            st.rerun()

        with st.expander("Offline options"):
            st.caption(
                "If the archive is unreachable, ExoScope stops rather than "
                "quietly substituting invented planets. You can opt into "
                "generated data for development — it is labelled everywhere "
                "it appears."
            )
            if st.button("Use synthetic data", use_container_width=True):
                st.cache_data.clear()
                ensure_data(force_refresh=True, allow_synthetic=True)
                st.rerun()

        st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
        model = st.session_state.get("model")
        if model is not None:
            result = st.session_state.get("train_result")
            f1 = (result.holdout_scores or result.cv_scores).get("f1", 0) if result else 0
            st.caption(
                f"🟢 {model.model_type} · {len(model.feature_names)} features  \n"
                f"held-out F1 **{f1:.3f}**"
            )
        else:
            st.caption("No model trained — open **Model Lab**")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="disclaimer">⚠️ Scores are a <b>likelihood index from '
            "proxy indicators</b> validated against Earth analogs — never a "
            "detection of life. False positives are the dominant historical "
            "risk in this field.</div>",
            unsafe_allow_html=True,
        )
    return mode


def render_masthead(mode: str) -> None:
    """Render the page masthead."""
    meta = st.session_state.get("meta")
    df = st.session_state.get("df")
    right = ""
    if meta is not None and df is not None:
        right = (
            f"<b>{meta.n_rows:,}</b> planets<br>"
            f"<b>{len(df.columns)}</b> engineered columns<br>"
            f"<b>{'XGBoost' if xgboost_available() else 'Random Forest'}</b> backend"
        )

    st.markdown(
        f'<div class="masthead"><div>'
        f'<h1>ExoScope <span class="accent">·</span> {mode}</h1>'
        f"<p>{MODES[mode]}. Physics-grounded habitability scoring over the full "
        f"NASA Exoplanet Archive parameter set, fused with vision-based "
        f"surface analysis and reported with calibrated uncertainty.</p>"
        f'</div><div class="masthead-meta">{right}</div></div>',
        unsafe_allow_html=True,
    )


def render_load_error() -> None:
    """Explain a failed archive fetch and offer the ways forward."""
    st.error("**Could not reach the NASA Exoplanet Archive.**")
    st.caption(st.session_state.load_error)
    st.markdown(
        "ExoScope fails loudly here on purpose. Earlier versions silently "
        "swapped in generated planets on a failed fetch and cached them to "
        "the same path as real data, so the whole pipeline could run on "
        "fabricated numbers with nothing in the UI to say so."
    )
    left, right = st.columns(2)
    with left:
        if st.button("↻ Retry", type="primary", use_container_width=True):
            st.cache_data.clear()
            ensure_data(force_refresh=True)
            st.rerun()
    with right:
        if st.button("Continue with synthetic data", use_container_width=True):
            st.cache_data.clear()
            ensure_data(force_refresh=True, allow_synthetic=True)
            st.rerun()


# ── Target Analysis ────────────────────────────────────────────────────
def planet_facts(row: pd.Series) -> str:
    """Build the key-value fact list for a planet."""
    facts = [
        ("Host star", row.get("hostname"), "{}"),
        ("Spectral class", row.get("spectral_class"), "{}"),
        ("Distance", row.get("sy_dist"), "{:.1f} pc"),
        ("Radius", row.get("pl_rade"), "{:.2f} R⊕"),
        ("Mass", row.get("pl_bmasse"), "{:.2f} M⊕"),
        ("Density", row.get("pl_dens"), "{:.2f} g/cm³"),
        ("Surface gravity", row.get("surface_gravity"), "{:.2f} g⊕"),
        ("Orbital period", row.get("pl_orbper"), "{:.2f} d"),
        ("Semi-major axis", row.get("pl_orbsmax"), "{:.4f} AU"),
        ("Eccentricity", row.get("pl_orbeccen"), "{:.3f}"),
        ("Insolation", row.get("pl_insol"), "{:.2f} S⊕"),
        ("Equilibrium temp", row.get("pl_eqt"), "{:.0f} K"),
        ("Discovered", row.get("disc_year"), "{:.0f}"),
        ("Method", row.get("discoverymethod"), "{}"),
        ("Facility", row.get("disc_facility"), "{}"),
    ]

    rows = []
    for label, value, fmt in facts:
        # Distinguish "not measured" from a real zero — the archive is
        # sparse, and an invented value is worse than an honest blank.
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            rows.append(f"<dt>{label}</dt><dd class='na'>not measured</dd>")
        else:
            try:
                rows.append(f"<dt>{label}</dt><dd>{fmt.format(value)}</dd>")
            except (ValueError, TypeError):
                rows.append(f"<dt>{label}</dt><dd>{value}</dd>")
    return f'<dl class="kv">{"".join(rows)}</dl>'


def attribution_list(factors: list, positive: bool) -> str:
    """Build the attribution rows for the contribution panels."""
    if not factors:
        return '<div class="note">No significant factors identified.</div>'

    largest = max((abs(v) for _, v, _ in factors), default=1.0) or 1.0
    css = "pos" if positive else "neg"
    rows = []
    for name, value, direction in factors[:6]:
        width = max(3, int(abs(value) / largest * 72))
        rows.append(
            f'<div class="factor {css}"><span class="name">{name}</span>'
            f'<span class="bar" style="width:{width}px"></span>'
            f'<span class="val">{value:+.4f}</span></div>'
        )
    return "".join(rows)


def render_target_analysis() -> None:
    """Single-planet scoring view."""
    df = st.session_state.df

    # ── Target picker ──────────────────────────────────────────────────
    controls = st.columns([3, 1.4, 1.4])
    with controls[0]:
        names = df["pl_name"].dropna().astype(str).tolist()
        # Default to a recognisable habitable-zone target rather than
        # whatever happens to sort first.
        default = next(
            (n for n in ("TRAPPIST-1 e", "Proxima Cen b", "TOI-700 d", "Kepler-452 b")
             if n in names), names[0] if names else None,
        )
        if default is None:
            st.warning("The loaded catalog has no named planets.")
            return
        planet = st.selectbox(
            "Target", names, index=names.index(default),
            help="Type to search all confirmed planets in the archive",
        )
    with controls[1]:
        only_scored = st.selectbox(
            "Shortlist", ["All planets", "Conservative HZ", "Optimistic HZ", "ESI > 0.7"],
            help="Narrow the picker to a physically interesting subset",
        )
    with controls[2]:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        analyze = st.button("Analyze target", type="primary", use_container_width=True)

    # Re-scope the selector when a shortlist is active.
    if only_scored != "All planets":
        subset = {
            "Conservative HZ": df[df.get("in_hz_conservative", 0) == 1],
            "Optimistic HZ": df[df.get("in_hz_optimistic", 0) == 1],
            "ESI > 0.7": df[df.get("esi", 0) > 0.7],
        }[only_scored]
        st.caption(
            f"**{len(subset)}** planets match *{only_scored}* — "
            f"{', '.join(subset['pl_name'].head(6).astype(str))}"
            + ("…" if len(subset) > 6 else "")
        )

    row = df[df["pl_name"] == planet].iloc[0]
    st.session_state.selected_planet = planet

    # ── Headline parameters ────────────────────────────────────────────
    section("Observed parameters", f"NASA Exoplanet Archive · {planet}")

    def fmt(value, spec="{:.2f}"):
        return "N/A" if value is None or pd.isna(value) else spec.format(value)

    hz_state = (
        ("Conservative HZ", "accent") if row.get("in_hz_conservative") == 1
        else ("Optimistic HZ", "") if row.get("in_hz_optimistic") == 1
        else ("Outside HZ", "")
    )
    metric_row([
        ("Radius", fmt(row.get("pl_rade")), "R⊕"),
        ("Mass", fmt(row.get("pl_bmasse")), "M⊕"),
        ("Equilibrium T", fmt(row.get("pl_eqt"), "{:.0f}"), "K"),
        ("Insolation", fmt(row.get("pl_insol")), "S⊕"),
        ("ESI", fmt(row.get("esi"), "{:.3f}"), "0-1", "primary"),
        ("Habitable zone", hz_state[0], "", hz_state[1]),
    ])

    # ── Score the target ───────────────────────────────────────────────
    model = st.session_state.get("model")
    if analyze and model is None:
        st.warning("Train a model first — open **Model Lab** in the sidebar.")
    if analyze and model is not None:
        with st.spinner("Scoring…"):
            prediction = model.predict(df[df["pl_name"] == planet])[0]
            fusion = FusionLayer()
            st.session_state[f"hbli::{planet}"] = (
                fusion.fuse(
                    target_name=prediction.planet_name,
                    tabular_score=prediction.habitability_score,
                    tabular_ci=(prediction.confidence_lower, prediction.confidence_upper),
                    tabular_features=dict(prediction.top_features + prediction.risk_factors),
                    has_tabular=True,
                    has_vision=False,  # no imagery in this view
                ),
                prediction,
            )

    stored = st.session_state.get(f"hbli::{planet}")
    if stored is not None:
        hbli, prediction = stored
        section("Habitability assessment", "tabular stream only — no imagery analyzed")

        cols = st.columns([1.1, 1, 1.3])
        with cols[0]:
            st.plotly_chart(
                charts.score_gauge(hbli.hbli_score,
                                   (hbli.confidence_lower, hbli.confidence_upper)),
                use_container_width=True, config={"displayModeBar": False},
            )
        with cols[1]:
            label, color, slug = score_band(hbli.hbli_score)
            st.markdown(
                f'<div class="panel"><h3>Verdict</h3>'
                f"{badge(hbli.category, slug)}"
                f'<p class="note" style="margin:0.9rem 0 0">{hbli.category_description}</p>'
                f'<div style="margin-top:1rem;font-family:var(--mono);font-size:0.78rem;'
                f'color:var(--muted)">95% CI '
                f"[{hbli.confidence_lower:.4f}, {hbli.confidence_upper:.4f}]</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.plotly_chart(
                charts.subscore_bars(hbli.tabular_score, hbli.vision_score,
                                     hbli.tabular_weight, hbli.vision_weight),
                use_container_width=True, config={"displayModeBar": False},
            )

        st.caption(
            "Only the tabular stream ran, so it carries the full weight. "
            "The vision stream is *absent*, not scored zero — an unanalyzed "
            "stream should not read as evidence against habitability."
        )

        # ── Attribution ────────────────────────────────────────────────
        section("Why this score?", "SHAP contributions from the trained model")
        attr = st.columns(2)
        with attr[0]:
            st.markdown(
                f'<div class="panel"><h3>Supporting factors</h3>'
                f"{attribution_list(hbli.top_contributing_factors, True)}</div>",
                unsafe_allow_html=True,
            )
        with attr[1]:
            st.markdown(
                f'<div class="panel"><h3>Risk factors</h3>'
                f"{attribution_list(hbli.risk_factors, False)}</div>",
                unsafe_allow_html=True,
            )

        if prediction.top_features or prediction.risk_factors:
            st.plotly_chart(
                charts.shap_waterfall(
                    prediction.top_features + prediction.risk_factors,
                    readable_name, prediction.habitability_score,
                ),
                use_container_width=True, config={"displayModeBar": False},
            )
    elif model is not None:
        st.info("Press **Analyze target** to score this planet.")

    # ── Context ────────────────────────────────────────────────────────
    section("Physical context", "how this target compares to Earth and the population")

    left, right = st.columns([1, 1.35])
    with left:
        st.markdown(
            f'<div class="panel"><h3>Archive record</h3>{planet_facts(row)}</div>',
            unsafe_allow_html=True,
        )
    with right:
        hz_fig = charts.habitable_zone_diagram(row)
        if hz_fig is not None:
            st.plotly_chart(hz_fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="panel"><h3>Habitable zone</h3>'
                '<p class="note">The host star falls outside the 2600–7200 K range '
                "where the Kopparapu et al. (2014) boundary fit is valid, so no "
                "habitable zone is computed for it. That is a limit of the model, "
                "not a statement about the planet.</p></div>",
                unsafe_allow_html=True,
            )

        radar = charts.earth_comparison_radar(row)
        if radar is not None:
            st.plotly_chart(radar, use_container_width=True,
                            config={"displayModeBar": False})

    st.plotly_chart(
        charts.population_scatter(
            df, "pl_insol", "pl_rade", "esi", highlight=planet,
            log_x=True, log_y=True, labels=READABLE,
        ),
        use_container_width=True,
    )
    st.caption(
        "Insolation versus radius for every confirmed planet, coloured by Earth "
        f"Similarity Index. **{planet}** is ringed in amber."
    )


# ── Catalog Explorer ───────────────────────────────────────────────────
def render_catalog_explorer() -> None:
    """Whole-archive exploration view."""
    df = st.session_state.df

    hab_count = int(df.get("in_hz_optimistic", pd.Series(dtype=float)).sum())
    cons_count = int(df.get("in_hz_conservative", pd.Series(dtype=float)).sum())
    nearby = int((df["sy_dist"] < 50).sum()) if "sy_dist" in df.columns else None
    esi_high = int((df["esi"] > 0.8).sum()) if "esi" in df.columns else None

    metric_row([
        ("Confirmed planets", f"{len(df):,}"),
        ("Optimistic HZ", f"{hab_count}", "Kopparapu 2014", "accent"),
        ("Conservative HZ", f"{cons_count}", "runaway → max greenhouse", "accent"),
        ("ESI > 0.8", f"{esi_high}" if esi_high is not None else "N/A"),
        ("Within 50 pc", f"{nearby:,}" if nearby is not None else "N/A"),
        ("Median ESI", f"{df['esi'].median():.3f}" if "esi" in df.columns else "N/A"),
    ])

    tabs = st.tabs([
        "Population", "Ranked targets", "Discovery history",
        "Correlations", "Data table",
    ])

    # ── Population ─────────────────────────────────────────────────────
    with tabs[0]:
        axis_options = {
            "pl_insol": "Insolation flux (S⊕)",
            "pl_rade": "Planet radius (R⊕)",
            "pl_bmasse": "Planet mass (M⊕)",
            "pl_orbper": "Orbital period (days)",
            "pl_eqt": "Equilibrium temperature (K)",
            "st_teff": "Stellar temperature (K)",
            "sy_dist": "Distance (pc)",
            "pl_dens": "Density (g/cm³)",
        }
        picker = st.columns([1, 1, 1, 1])
        with picker[0]:
            x = st.selectbox("X axis", list(axis_options), index=0,
                             format_func=lambda c: axis_options[c])
        with picker[1]:
            y = st.selectbox("Y axis", list(axis_options), index=1,
                             format_func=lambda c: axis_options[c])
        with picker[2]:
            color_options = {"esi": "Earth Similarity Index",
                             "pl_eqt": "Equilibrium temperature",
                             "st_teff": "Stellar temperature",
                             "hz_position": "Position across the HZ"}
            color = st.selectbox("Colour by", list(color_options), index=0,
                                 format_func=lambda c: color_options[c])
        with picker[3]:
            log_axes = st.multiselect("Log scale", ["X", "Y"], default=["X", "Y"])

        st.plotly_chart(
            charts.population_scatter(
                df, x, y, color, log_x="X" in log_axes, log_y="Y" in log_axes,
                labels={**axis_options, **color_options, **READABLE},
            ),
            use_container_width=True,
        )

    # ── Ranked targets ─────────────────────────────────────────────────
    with tabs[1]:
        model = st.session_state.get("model")
        if model is None:
            st.info(
                "Train a model in **Model Lab** to rank every planet by "
                "calibrated habitability score. Ranked by ESI in the meantime."
            )
            ranked = df.nlargest(50, "esi") if "esi" in df.columns else df.head(50)
            display_cols = [c for c in ("pl_name", "hostname", "esi", "pl_rade", "pl_eqt",
                                        "sy_dist", "spectral_class", "discoverymethod")
                            if c in ranked.columns]
            st.dataframe(
                ranked[display_cols].rename(columns=READABLE),
                use_container_width=True, height=560, hide_index=True,
            )
        else:
            if st.session_state.get("catalog_scores") is None:
                with st.spinner("Scoring the full catalog…"):
                    st.session_state.catalog_scores = model.score_catalog(df)

            scores = st.session_state.catalog_scores
            top_n = st.slider("Show top N targets", 10, 200, 40, 10)
            shown = scores.head(top_n).copy()
            shown.insert(0, "Rank", range(1, len(shown) + 1))

            st.dataframe(
                shown.rename(columns={
                    "pl_name": "Planet", "hostname": "Host",
                    "habitability_score": "Score", "ci_lower": "CI low",
                    "ci_upper": "CI high", "esi": "ESI", "sy_dist": "Distance (pc)",
                    "spectral_class": "Class", "category": "Category",
                    "pl_rade": "Radius (R⊕)", "pl_eqt": "T_eq (K)",
                    "discoverymethod": "Method", "disc_year": "Year",
                    "in_hz_conservative": "Cons. HZ", "in_hz_optimistic": "Opt. HZ",
                }),
                use_container_width=True, height=560, hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                },
            )
            st.download_button(
                "Download ranked targets (CSV)",
                scores.to_csv(index=False).encode("utf-8"),
                file_name="exoscope_ranked_targets.csv",
                mime="text/csv",
            )

    # ── Discovery history ──────────────────────────────────────────────
    with tabs[2]:
        st.plotly_chart(charts.discovery_timeline(df), use_container_width=True)
        st.caption(
            "Discovery method is a genuine model feature, not just bookkeeping: "
            "it encodes observational selection. Transit surveys favour "
            "short-period planets around bright nearby stars; microlensing "
            "finds distant ones that will never be characterised further."
        )

        breakdown = st.columns(2)
        for column, (source, heading, axis) in zip(
            breakdown,
            (("discoverymethod", "By method", "Method"),
             ("disc_facility", "By facility", "Facility")),
        ):
            if source not in df.columns:
                continue
            counts = df[source].value_counts().head(8)
            with column, panel(heading):
                st.dataframe(
                    counts.rename("Planets").rename_axis(axis).reset_index(),
                    use_container_width=True, hide_index=True, height=310,
                )

    # ── Correlations ───────────────────────────────────────────────────
    with tabs[3]:
        candidates = [
            "pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbeccen",
            "pl_insol", "pl_eqt", "st_teff", "st_mass", "st_rad", "st_met",
            "sy_dist", "esi", "hz_position", "tidal_lock_likelihood",
            "stellar_activity", "surface_gravity",
        ]
        chosen = st.multiselect(
            "Features", [c for c in candidates if c in df.columns],
            default=[c for c in candidates[:12] if c in df.columns],
            format_func=readable_name,
        )
        if len(chosen) >= 2:
            st.plotly_chart(
                charts.correlation_heatmap(df, chosen, readable_name),
                use_container_width=True,
            )
        else:
            st.info("Select at least two features.")

    # ── Data table ─────────────────────────────────────────────────────
    with tabs[4]:
        raw = st.session_state.raw_df
        search = st.text_input("Filter by planet or host name", placeholder="e.g. TRAPPIST")
        table = raw
        if search:
            mask = (
                raw["pl_name"].astype(str).str.contains(search, case=False, na=False)
                | raw["hostname"].astype(str).str.contains(search, case=False, na=False)
            )
            table = raw[mask]
            st.caption(f"{len(table):,} matching planets")

        st.dataframe(table, use_container_width=True, height=560, hide_index=True)
        st.download_button(
            "Download the NASA table (CSV)",
            raw.to_csv(index=False).encode("utf-8"),
            file_name="nasa_exoplanet_archive.csv",
            mime="text/csv",
        )


# ── Entry point ────────────────────────────────────────────────────────
def main() -> None:
    """Render the dashboard."""
    # Load before drawing any chrome, so the sidebar and masthead can report
    # the real row count and provenance rather than "Not loaded".
    data_ok = ensure_data()

    mode = render_sidebar()
    render_masthead(mode)

    if mode == "Surface Imagery":
        # The vision pipeline is independent of the exoplanet catalog, so it
        # stays usable even when the archive is unreachable.
        image_analysis.render()
    else:
        if not data_ok:
            render_load_error()
            st.stop()

        provenance_banner(st.session_state.meta)

        if mode == "Target Analysis":
            render_target_analysis()
        elif mode == "Catalog Explorer":
            render_catalog_explorer()
        elif mode == "Model Lab":
            model_lab.render()

    st.markdown(
        '<div class="appfooter">'
        "ExoScope v0.2 — Habitability &amp; Biosignature-Likelihood Index<br>"
        "Data: NASA Exoplanet Archive (<code>pscomppars</code>). "
        "Habitable zone boundaries: Kopparapu et al. (2014). "
        "ESI: Schulze-Makuch et al. (2011).<br>"
        "All scores are proxy-based likelihood estimates, not detections of life."
        "</div>",
        unsafe_allow_html=True,
    )


main()
