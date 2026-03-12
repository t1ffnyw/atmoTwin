import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_dashboard_root = Path(__file__).resolve().parent.parent
_project_root = _dashboard_root.parent

for _p in (_dashboard_root, _project_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from components.result_cards import render_upload_classification
from components.spectrum_plot import make_contributions_figure, make_spectrum_figure
from state import init_state
from ui import configure_page

configure_page()
init_state()

st.title("Spectrum Viewer")

tab_sim, tab_upload = st.tabs(["Simulated Spectrum", "Upload CSV"])

# ── Tab 1: Simulated Spectrum (existing behaviour) ─────────────────────────────

with tab_sim:
    if st.session_state.spectrum:
        show_bands = st.toggle("Highlight biosignature bands", value=True)
        fig = make_spectrum_figure(st.session_state.spectrum, show_bands)
        st.plotly_chart(fig, width="stretch")

        if st.session_state.get("contributions"):
            st.divider()
            st.subheader("Per-molecule contributions")
            st.caption(
                "Each colored area shows how much a single molecule contributes "
                "to the overall spectrum. Computed by subtracting a spectrum with "
                "that molecule removed from the baseline."
            )
            c = st.session_state.contributions
            fig_c = make_contributions_figure(c["wavelength"], c["molecules"])
            st.plotly_chart(fig_c, width="stretch")
    else:
        st.info("No spectrum yet. Run a simulation from the Builder page.")


# ── Tab 2: Upload CSV ──────────────────────────────────────────────────────────

with tab_upload:
    st.markdown(
        "Upload a two-column CSV file with **wavelength (µm)** and **flux** values. "
        "The spectrum must cover **4.0 – 18.33 µm**. "
        "A header row is optional — any column names are accepted."
    )

    with st.expander("Example CSV format"):
        st.code(
            "wavelength,flux\n"
            "4.00,0.11647\n"
            "4.50,0.07553\n"
            "5.00,0.04921\n"
            "...\n"
            "18.33,0.00044",
            language="text",
        )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        # ── parse ──────────────────────────────────────────────────────────────
        try:
            # Try with header first; fall back to headerless
            raw = uploaded.read()
            uploaded.seek(0)

            df = pd.read_csv(uploaded)
            # If column names look like numbers the file probably has no header
            first_col_name = str(df.columns[0])
            try:
                float(first_col_name)
                # Numeric column name → re-read without header
                import io
                df = pd.read_csv(io.BytesIO(raw), header=None)
            except ValueError:
                pass

            if df.shape[1] < 2:
                st.error("CSV must have at least two columns (wavelength and flux).")
                st.stop()

            wavelengths_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            flux_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()

            nan_mask = np.isnan(wavelengths_raw) | np.isnan(flux_raw)
            if nan_mask.sum() > 0:
                wavelengths_raw = wavelengths_raw[~nan_mask]
                flux_raw = flux_raw[~nan_mask]

            if len(wavelengths_raw) < 20:
                st.error(
                    f"Too few data points ({len(wavelengths_raw)}). "
                    "Upload a CSV with at least 20 rows."
                )
                st.stop()

            # Sort by wavelength in case the file isn't ordered
            order = np.argsort(wavelengths_raw)
            wavelengths_raw = wavelengths_raw[order]
            flux_raw = flux_raw[order]

        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")
            st.stop()

        # ── plot uploaded spectrum ─────────────────────────────────────────────
        st.subheader("Uploaded Spectrum")
        show_bands_upload = st.toggle("Highlight biosignature bands", value=True, key="upload_bands")
        spectrum_dict = {"wavelength": wavelengths_raw, "depth": flux_raw}
        fig_up = make_spectrum_figure(spectrum_dict, show_bands_upload)
        st.plotly_chart(fig_up, width="stretch")

        st.divider()

        # ── classify ───────────────────────────────────────────────────────────
        st.subheader("AtmoTwin Classification")

        if st.button("Analyse with AtmoTwin", type="primary"):
            with st.spinner("Running Random Forest classifier…"):
                try:
                    from model.inference import predict
                    result = predict(wavelengths_raw, flux_raw)
                    st.session_state["_upload_classification"] = result
                except ValueError as exc:
                    st.error(str(exc))
                    st.session_state.pop("_upload_classification", None)
                except Exception as exc:
                    st.error(f"Classification failed: {exc}")
                    st.session_state.pop("_upload_classification", None)

        if st.session_state.get("_upload_classification"):
            render_upload_classification(st.session_state["_upload_classification"])
        else:
            st.caption("Press the button above to classify your uploaded spectrum.")
