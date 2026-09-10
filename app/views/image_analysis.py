"""
Surface Imagery — classical CV pre-filter plus the YOLO specialist stack.

Runs the vision half of the pipeline over uploaded planetary surface
imagery: tile, pre-filter, detect, then aggregate detections into a
biosignature-proxy score.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app import charts
from app.theme import COLORS, badge, disclaimer, metric_row, panel, section
from src.data_ingestion.tiling import tile_image
from src.models.fusion import FusionLayer
from src.models.vision_detector import VisionDetectorStack
from src.preprocessing.classical_cv_filter import ClassicalCVFilter


def render() -> None:
    """Render the Surface Imagery view."""
    detector = _get_detector()
    loaded = [name for name, info in detector.specialists.items() if info.is_loaded]

    if not loaded:
        st.markdown(
            '<div class="disclaimer">⚠️ <b>No trained YOLO weights found.</b> '
            "The detector stack is running in <b>simulation mode</b>: detections "
            "below are derived from image texture statistics, not from a trained "
            "biosignature model. They demonstrate the pipeline end to end, but "
            "carry no evidential weight. Train the specialists and place the "
            "weights in <code>models/</code> to enable real inference."
            "</div>",
            unsafe_allow_html=True,
        )

    section("Input", "HiRISE, Mastcam, or Earth-analog surface imagery")

    upload = st.file_uploader(
        "Upload a planetary surface image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="The image is tiled, pre-filtered, and scanned for geomorphological "
             "biosignature proxies.",
    )

    if upload is None:
        _render_placeholder(detector)
        return

    image = Image.open(upload).convert("RGB")
    array = np.array(image)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    preview, settings = st.columns([1.5, 1])
    with preview:
        st.image(
            image,
            caption=f"{upload.name} — {array.shape[1]}×{array.shape[0]} px",
            use_container_width=True,
        )
    with settings:
        with panel("Analysis settings"):
            tile_size = st.select_slider(
                "Tile size", options=[256, 320, 448, 512, 640, 768, 1024], value=640,
                help="640 px matches the YOLO input size.",
            )
            threshold = st.slider("Detection confidence threshold", 0.05, 0.90, 0.25, 0.05)
            max_preview = st.slider("Detection tiles to display", 3, 12, 6, 3)

        estimated = max(1, (array.shape[0] // tile_size) * (array.shape[1] // tile_size))
        st.caption(f"≈ {estimated} tiles at {tile_size} px")

        run = st.button("Run vision pipeline", type="primary", use_container_width=True)

    if not run:
        return

    progress = st.progress(0, "Tiling image…")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "upload.png"
        cv2.imwrite(str(temp_path), bgr)

        tiling = tile_image(str(temp_path), tile_size=tile_size)

        progress.progress(30, "Running the classical CV pre-filter…")
        cv_filter = ClassicalCVFilter()
        passed_tiles, rejected_tiles, filter_summary = cv_filter.filter_batch(tiling.tiles)

        progress.progress(60, "Scanning for biosignature proxies…")
        results = detector.detect_batch(passed_tiles)

        progress.progress(90, "Aggregating…")

        _render_results(
            detector, tiling, filter_summary, passed_tiles, results,
            upload.name, threshold, max_preview,
        )
        progress.progress(100, "Complete")


@st.cache_resource(show_spinner=False)
def _get_detector() -> VisionDetectorStack:
    """Load the detector stack once per session."""
    return VisionDetectorStack()


def _render_results(detector, tiling, filter_summary, passed_tiles, results,
                    source_name: str, threshold: float, max_preview: int) -> None:
    """Render pipeline metrics, detection previews and the fused score."""
    total_detections = sum(
        1 for r in results for d in r.detections if d.confidence >= threshold
    )
    mean_score = float(np.mean([r.overall_score for r in results])) if results else 0.0
    rejected = filter_summary["total"] - filter_summary["passed"]
    saved = rejected / max(filter_summary["total"], 1)

    section("Pipeline", "tiling → classical pre-filter → specialist detection")
    metric_row([
        ("Tiles generated", f"{tiling.total_tiles:,}", f"{tiling.tile_size} px"),
        ("Passed pre-filter", f"{filter_summary['passed']:,}",
         f"{filter_summary['pass_rate']:.0%} of tiles"),
        ("Compute saved", f"{saved:.0%}", f"{rejected:,} tiles skipped", "accent"),
        ("Mean relevance", f"{filter_summary['mean_relevance']:.3f}"),
        ("Detections", f"{total_detections:,}", f"≥ {threshold:.2f} confidence"),
        ("Vision score", f"{mean_score:.3f}", "0-1", "primary"),
    ])

    # ── Detection previews ─────────────────────────────────────────────
    if results:
        ranked = sorted(
            range(len(results)), key=lambda i: results[i].overall_score, reverse=True
        )[:max_preview]

        if ranked and results[ranked[0]].overall_score > 0:
            section("Highest-scoring tiles", "bounding boxes drawn over the source tiles")
            columns = st.columns(3)
            for slot, index in enumerate(ranked):
                result = results[index]
                tile = passed_tiles[index]
                overlay = detector.visualize_detections(tile.image, result)
                with columns[slot % 3]:
                    st.image(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        caption=f"Score {result.overall_score:.3f} · "
                                f"{result.n_detections} detection(s)",
                        use_container_width=True,
                    )

        class_counts: dict[str, int] = {}
        for result in results:
            for detection in result.detections:
                if detection.confidence >= threshold:
                    class_counts[detection.class_name] = (
                        class_counts.get(detection.class_name, 0) + 1
                    )

        if class_counts:
            st.plotly_chart(
                charts.detection_class_bars(class_counts),
                use_container_width=True, config={"displayModeBar": False},
            )

    # ── Fused score ────────────────────────────────────────────────────
    section("Vision-only HBLI", "no exoplanet parameters supplied for this target")

    # Take the strongest confidence per class as that class's feature value.
    vision_features = {}
    for class_name in detector.specialists:
        confidences = [
            d.confidence for r in results for d in r.detections
            if d.class_name == class_name and d.confidence >= threshold
        ]
        if confidences:
            vision_features[class_name] = max(confidences)

    hbli = FusionLayer().fuse(
        target_name=source_name,
        tabular_score=0.0,
        vision_score=mean_score,
        vision_features=vision_features,
        has_tabular=False,   # no planetary parameters in this view
        has_vision=True,
    )

    left, right = st.columns([1, 1.4])
    with left:
        st.plotly_chart(
            charts.score_gauge(hbli.hbli_score,
                               (hbli.confidence_lower, hbli.confidence_upper),
                               title="Vision HBLI"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with right:
        _, _, slug = _band(hbli.hbli_score)
        st.markdown(
            f'<div class="panel"><h3>Interpretation</h3>{badge(hbli.category, slug)}'
            f'<p class="note" style="margin:0.9rem 0 0">{hbli.category_description}</p>'
            f'<p class="note" style="margin-top:0.9rem">This is a <b>vision-only</b> '
            f"score: the tabular stream contributed nothing because no planetary "
            f"parameters were supplied, so the vision stream carries the full "
            f"weight. Pair it with a target in <b>Target Analysis</b> for a "
            f"complete assessment.</p></div>",
            unsafe_allow_html=True,
        )

    disclaimer(
        "Surface morphology is a <b>proxy</b>, not a biosignature. Abiotic "
        "processes produce layering, evaporites and dendritic channels; "
        "false positives are the dominant historical failure mode in this "
        "field. Detections mark targets worth further observation — nothing more."
    )


def _band(score: float):
    """Local import shim so the theme stays the single source of score bands."""
    from app.theme import score_band
    return score_band(score)


def _render_placeholder(detector: VisionDetectorStack) -> None:
    """Explain the pipeline while no image is loaded."""
    st.info("Upload a surface image to begin.")

    left, right = st.columns(2)
    with left, panel("How the pipeline works"):
        st.markdown(
            "1. **Tile** — the image is split into overlapping tiles at the "
            "detector's native input size.\n"
            "2. **Pre-filter** — a classical CV pass scores each tile on edge "
            "density, local entropy and colour variance, discarding featureless "
            "or no-data tiles before any deep model runs. This is where most of "
            "the compute saving comes from.\n"
            "3. **Detect** — specialist YOLO models scan the surviving tiles, "
            "one model per proxy class.\n"
            "4. **Aggregate** — per-class confidences are pooled into a single "
            "vision score and fused into the HBLI."
        )

    with right, panel("Biosignature proxy classes"):
        for name, info in detector.specialists.items():
            state = "loaded" if info.is_loaded else "simulated"
            variant = "high" if info.is_loaded else "neutral"
            st.markdown(
                f"<div style='margin-bottom:0.85rem'>"
                f"<b>{name.replace('_', ' ').title()}</b> {badge(state, variant)}"
                f"<div class='note'>{info.description}</div></div>",
                unsafe_allow_html=True,
            )
