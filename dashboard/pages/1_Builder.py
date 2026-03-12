import sys
from pathlib import Path

import streamlit as st

# Ensure dashboard root is on path so "psg" resolves when run via streamlit run dashboard/Home.py
_dashboard_root = Path(__file__).resolve().parent.parent
if str(_dashboard_root) not in sys.path:
    sys.path.insert(0, str(_dashboard_root))

from config import SCENARIO_PRESETS
from state import init_state, get_planet_params, load_preset
from components.planet_controls import render_gas_sliders
from components.spectrum_plot import make_spectrum_figure
from components.result_cards import (
    render_classification_card,
    render_false_positive_warnings,
)
from psg.service import generate_spectrum
from ui import configure_page


configure_page()
init_state()

st.title("Atmosphere Builder")
st.caption(
    "Configure a planet-star system, run the spectrum simulation, and inspect the classification."
)

st.divider()

# Sidebar: preset loader
def _on_preset_change():
    name = st.session_state.preset_selector
    if name != "(custom)":
        load_preset(name)

with st.sidebar:
    st.header("Quick Start")
    st.selectbox(
        "Load scenario preset",
        ["(custom)"] + list(SCENARIO_PRESETS.keys()),
        key="preset_selector",
        on_change=_on_preset_change,
    )

# Main layout: spectrum/planet on left, planet parameters on right
col_display, col_controls = st.columns([2, 1])

# Render controls first so run_clicked is defined before display block uses it
with col_controls:
    render_gas_sliders()

    st.divider()
    run_clicked = st.button("🚀 Simulate", use_container_width=True, type="primary")

with col_display:
    if run_clicked:
        with st.spinner("Generating spectrum via PSG..."):
            try:
                params = get_planet_params()
                st.session_state.spectrum = generate_spectrum(params, output_type="rad")
                # Placeholder classifier until ML pipeline is wired
                st.session_state.classification = {
                    "label": "disequilibrium",
                    "confidence": 0.87,
                    "key_features": [
                        ("O₂/CH₄ ratio", 0.42),
                        ("CH₄ band depth (3.3μm)", 0.28),
                        ("CO absence score", 0.18),
                    ],
                }
                st.session_state.false_positive_flags = []
            except FileNotFoundError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"PSG error: {e}")

    # Display results if available
    if st.session_state.spectrum:
        tab_spec, tab_class = st.tabs(["📈 Spectrum", "🧬 Classification"])

        with tab_spec:
            show_bands = st.toggle("Highlight biosignature bands", value=True)
            fig = make_spectrum_figure(st.session_state.spectrum, show_bands)
            st.plotly_chart(fig, use_container_width=True)

        with tab_class:
            if st.session_state.classification:
                render_classification_card(st.session_state.classification)
                st.divider()
                render_false_positive_warnings(st.session_state.false_positive_flags)
    else:
        st.info(
            "Configure your planet on the right and hit **Simulate** to generate a spectrum."
        )

