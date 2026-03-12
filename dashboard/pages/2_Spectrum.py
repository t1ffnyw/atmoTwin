import sys
from pathlib import Path

import streamlit as st

_dashboard_root = Path(__file__).resolve().parent.parent
if str(_dashboard_root) not in sys.path:
    sys.path.insert(0, str(_dashboard_root))

from components.spectrum_plot import make_spectrum_figure, make_contributions_figure
from state import init_state
from ui import configure_page


configure_page()
init_state()

st.title("Spectrum Viewer")

if st.session_state.spectrum:
    show_bands = st.toggle("Highlight biosignature bands", value=True)
    fig = make_spectrum_figure(st.session_state.spectrum, show_bands)
    st.plotly_chart(fig, width='stretch')

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
        st.plotly_chart(fig_c, width='stretch')
else:
    st.info("No spectrum yet. Run a simulation from the Builder page.")
