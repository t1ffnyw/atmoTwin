import streamlit as st

from components.spectrum_plot import make_spectrum_figure
from state import init_state
from ui import configure_page


configure_page()
init_state()

st.title("Spectrum Viewer")

if st.session_state.spectrum:
    show_bands = st.toggle("Highlight biosignature bands", value=True)
    fig = make_spectrum_figure(st.session_state.spectrum, show_bands)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No spectrum yet. Run a simulation from the main AtmoTwin page.")

