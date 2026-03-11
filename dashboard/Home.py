import streamlit as st

from state import init_state
from ui import configure_page


configure_page()
init_state()

st.title("Home")
st.caption("Welcome to AtmoTwin — a digital twin for exploring exoplanet atmospheric chemistry and biosignature detectability.")

st.divider()

st.subheader("What you can do here")
st.markdown(
    """
- **Builder**: Configure host stars, orbits, and atmospheric compositions, then generate simulated spectra.
- **Spectrum**: Inspect the transmission spectrum, with biosignature bands highlighted.
- **Classifier**: View ML-based classifications of atmospheric disequilibrium and contributing features.
- **Explorer**: Save and compare multiple scenarios or presets side by side.
"""
)

st.subheader("Suggested workflow")
st.markdown(
    """
1. Go to **Builder** and start with a preset or custom configuration.
2. Click **Simulate** to generate a spectrum and classification.
3. Use the **Spectrum** and **Classifier** pages to dive into the results.
4. Save interesting cases in **Explorer** to compare different scenarios.
"""
)

st.info("Use the sidebar to switch between pages at any time.")

