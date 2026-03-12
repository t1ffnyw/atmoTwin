import sys
from pathlib import Path

import numpy as np
import streamlit as st

_dashboard_root = Path(__file__).resolve().parent.parent
_project_root = _dashboard_root.parent

for _p in (_dashboard_root, _project_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from components.result_cards import (
    render_classification_card,
    render_false_positive_warnings,
    render_upload_classification,
)
from components.spectrum_plot import make_contributions_figure, make_spectrum_figure
from state import init_state
from ui import configure_page


configure_page()
init_state()

st.title("Results")

tab_spec, tab_class = st.tabs(["Spectrum", "Classification"])


# ── Tab 1: Spectrum visualisation ──────────────────────────────────────────────

with tab_spec:
    sim_spec = st.session_state.get("spectrum")
    uploaded = st.session_state.get("uploaded_builder_spectrum")

    if not sim_spec and not uploaded:
        st.info("No spectrum available yet. Run a simulation or upload a spectrum from the Builder page.")
    else:
        source_options = []
        if sim_spec:
            source_options.append("Simulated (Builder)")
        if uploaded:
            source_options.append("Uploaded (CSV from Builder)")

        if len(source_options) > 1:
            source_label = st.radio(
                "Select spectrum source",
                options=source_options,
                horizontal=True,
                key="results_spectrum_source",
            )
        else:
            source_label = source_options[0]

        use_sim = source_label.startswith("Simulated")

        if use_sim:
            spectrum_dict = sim_spec
        else:
            spectrum_dict = {
                "wavelength": np.asarray(uploaded["wavelength"]),
                "depth": np.asarray(uploaded["flux"]),
            }

        st.subheader("Spectrum")
        show_bands = st.toggle("Highlight biosignature bands", value=True)

        fig = make_spectrum_figure(spectrum_dict, show_bands)
        st.plotly_chart(fig, width="stretch")

        if use_sim and st.session_state.get("contributions"):
            st.divider()
            st.subheader("Per-molecule contributions")
            st.caption(
                "Each colored curve shows how much a single molecule contributes "
                "to the overall spectrum, computed by removing that molecule from the model."
            )
            c = st.session_state.contributions
            fig_c = make_contributions_figure(c["wavelength"], c["molecules"])
            st.plotly_chart(fig_c, width="stretch")
        elif not use_sim:
            st.info(
                "Per-molecule contribution curves are only available for simulated spectra "
                "generated from the Atmosphere Composer."
            )


# ── Tab 2: Classifier output ───────────────────────────────────────────────────

with tab_class:
    sim_class = st.session_state.get("classification")
    upload_spec = st.session_state.get("uploaded_builder_spectrum")

    if not sim_class and not st.session_state.get("_upload_classification") and not upload_spec:
        st.info(
            "No classification results available yet. Run a simulation or upload a spectrum "
            "from the Builder page first."
        )
    else:
        if sim_class:
            st.subheader("Classification for simulated spectrum")
            render_upload_classification(sim_class)
            st.divider()
            render_false_positive_warnings(st.session_state.get("false_positive_flags", []))

        if upload_spec:
            st.subheader("Classification for uploaded spectrum")
            if st.button("Run AtmoTwin classifier on uploaded spectrum", type="primary"):
                with st.spinner("Running Random Forest classifier on uploaded spectrum…"):
                    try:
                        from model.inference import predict

                        result = predict(
                            np.asarray(upload_spec["wavelength"]),
                            np.asarray(upload_spec["flux"]),
                        )
                        st.session_state["_upload_classification"] = result
                    except Exception as exc:
                        st.error(f"Classification failed: {exc}")
                        st.session_state.pop("_upload_classification", None)

            if st.session_state.get("_upload_classification"):
                render_upload_classification(st.session_state["_upload_classification"])

    st.divider()
    st.markdown("#### How to interpret these classifications")
    st.markdown(
        """
The AtmoTwin classifier compares your spectrum against a library of simulated
planet–star systems and assigns it to one of several regimes (e.g., inhabited
Modern Earth–like, Archean-like, lifeless, or false positive).\n
The probabilities reflect how consistently the spectrum matches each regime in
feature space (e.g., band depths and ratios of O₃, CH₄, CO₂, H₂O, N₂O, CO).\n
Use these labels as **diagnostic guidance**, not as definitive detections:
they depend on the assumed forward models, noise properties, and the training
set. For publication-grade analysis you should always cross-check key molecular
bands, instrumental systematics, and astrophysical false-positive scenarios.
        """
    )