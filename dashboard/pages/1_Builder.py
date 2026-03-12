import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# Ensure dashboard root is on path so "psg" resolves when run via streamlit run dashboard/Home.py
_dashboard_root = Path(__file__).resolve().parent.parent
_project_root = _dashboard_root.parent

for _p in (_dashboard_root, _project_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import SCENARIO_PRESETS
from state import init_state, get_planet_params, load_preset
from components.planet_controls import render_gas_inputs
from components.spectrum_plot import make_spectrum_figure
from components.result_cards import (
    render_false_positive_warnings,
    render_upload_classification,
)
from psg.service import generate_spectrum, generate_model_spectrum, calculate_contributions
from ui import configure_page


configure_page()
init_state()


st.title("Atmosphere Builder")
st.caption(
    "Either compose an atmosphere from presets and gases or upload an observed spectrum, "
    "then run the pipeline and inspect results on the Results page."
)

st.divider()


def _set_preset(name: str) -> None:
    """Apply a scenario preset and track it for 'Custom' labelling."""
    load_preset(name)
    st.session_state.current_preset = name


def _current_composition_label() -> str:
    """Return a user-facing label describing whether we're on a preset or custom mix."""
    base = st.session_state.get("current_preset", "(custom)")
    if base not in SCENARIO_PRESETS:
        return "Current composition: Custom"

    preset_gases = SCENARIO_PRESETS[base]["gases"]
    gases = st.session_state.get("gases", {})

    # If any gas differs from the preset by more than a tiny tolerance, treat as custom
    diverged = False
    for gas_id, preset_val in preset_gases.items():
        current_val = gases.get(gas_id)
        if current_val is None:
            diverged = True
            break
        if abs(float(current_val) - float(preset_val)) > 1e-6:
            diverged = True
            break

    if diverged:
        return f"Current composition: Custom (modified from {base})"
    return f"Current composition: {base}"


def _render_preset_row() -> None:
    st.markdown("**Scenario presets**")
    cols = st.columns(len(SCENARIO_PRESETS))
    for col, name in zip(cols, SCENARIO_PRESETS.keys()):
        with col:
            if st.button(name, key=f"preset_btn_{name}"):
                _set_preset(name)

    st.caption(_current_composition_label())


def _parse_builder_csv(uploaded) -> Dict[str, Any] | None:
    """Parse and validate a CSV with wavelength, flux, error columns."""
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return None

    expected_cols = ["wavelength", "flux", "error"]
    cols_lower = [str(c).strip().lower() for c in df.columns]
    if cols_lower != expected_cols:
        st.error(
            "CSV must have exactly three columns named "
            "`wavelength`, `flux`, and `error` in that order."
        )
        return None

    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[expected_cols].isna().any().any():
        st.error("All values in wavelength, flux, and error columns must be numeric.")
        return None

    if len(df) < 10:
        st.error("CSV must have at least 10 rows.")
        return None

    # Sort by wavelength
    df = df.sort_values("wavelength")

    data = {
        "wavelength": df["wavelength"].to_numpy(dtype=float),
        "flux": df["flux"].to_numpy(dtype=float),
        "error": df["error"].to_numpy(dtype=float),
        "preview": df.head(10),
    }
    return data


tab_composer, tab_upload = st.tabs(["Atmosphere Composer", "Upload Spectrum"])


# ── Tab 1: Atmosphere Composer ──────────────────────────────────────────────────

with tab_composer:
    st.subheader("Atmosphere Composer")
    st.markdown(
        "Start from a curated preset (Modern Earth, Archean Earth, etc.) and then fine‑tune "
        "the atmospheric gas abundances. Host star and orbit are fixed for this demo."
    )

    _render_preset_row()

    render_gas_inputs()

    st.divider()
    run_clicked = st.button("Run with atmosphere", type="primary")

    if run_clicked:
        # Run the PSG forward model and classifier, as before
        with st.spinner("Generating spectrum and molecule contributions via PSG..."):
            try:
                params = get_planet_params()
                st.session_state.spectrum = generate_spectrum(params, output_type="rad")

                w, y_base, contribs = calculate_contributions(params, output_type="rad")
                st.session_state.contributions = {
                    "wavelength": w,
                    "baseline": y_base,
                    "molecules": contribs,
                }

                from model.inference import predict

                model_spec = generate_model_spectrum(params, output_type="rad")
                st.session_state.classification = predict(
                    model_spec["wavelength"],
                    model_spec["depth"],
                )
                st.session_state.false_positive_flags = []
                st.session_state.builder_source = "composition"
            except FileNotFoundError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"PSG error: {e}")

    # Quick link to the Spectrum (Results) page
    if st.session_state.get("spectrum"):
        st.success(
            "Simulation complete. Open the Results page to inspect spectra and classifications."
        )
        st.page_link("pages/2_Results.py", label="Go to Results")


# ── Tab 2: CSV Upload ───────────────────────────────────────────────────────────

with tab_upload:
    st.subheader("Upload observed / external spectrum")
    st.markdown(
        "Upload a `.csv` file with three columns: **wavelength**, **flux**, **error**. "
        "This will be stored in app state for downstream analysis on the Results page."
    )

    with st.expander("Required CSV format", expanded=False):
        st.code(
            "wavelength,flux,error\n"
            "4.00,0.11647,0.002\n"
            "4.50,0.07553,0.002\n"
            "5.00,0.04921,0.0015\n"
            "...\n"
            "18.33,0.00044,0.0002",
            language="text",
        )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="builder_csv")

    if uploaded is not None:
        parsed = _parse_builder_csv(uploaded)
        if parsed is not None:
            # Store in shared app state for the Results page
            st.session_state.uploaded_builder_spectrum = {
                "wavelength": parsed["wavelength"],
                "flux": parsed["flux"],
                "error": parsed["error"],
            }
            st.session_state.builder_source = "upload"

            st.success("CSV parsed successfully and stored in the Builder state.")

            st.markdown("**Preview (first 10 rows)**")
            st.dataframe(parsed["preview"])

            st.markdown("**Quick-look spectrum**")
            # Reuse the spectrum plotting helper for a consistent look
            spectrum_dict = {
                "wavelength": parsed["wavelength"],
                "depth": parsed["flux"],
            }
            show_bands = st.toggle(
                "Highlight biosignature bands",
                value=True,
                key="builder_upload_bands",
            )
            fig = make_spectrum_figure(spectrum_dict, show_bands)
            st.plotly_chart(fig, width="stretch")

    st.divider()
    run_upload = st.button("Use uploaded spectrum", type="primary")

    if run_upload:
        if st.session_state.get("uploaded_builder_spectrum") is None:
            st.error(
                "No valid spectrum uploaded yet. Please upload a CSV that passes validation first."
            )
        else:
            st.success(
                "Uploaded spectrum is ready. Open the Results page to run analyses."
            )
            st.page_link("pages/2_Results.py", label="Go to Results")