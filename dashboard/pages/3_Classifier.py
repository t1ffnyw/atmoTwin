import streamlit as st

from components.result_cards import (
    render_classification_card,
    render_false_positive_warnings,
)
from state import init_state
from ui import configure_page


configure_page()
init_state()

st.title("Classifier Results")

classification = st.session_state.classification

if classification:
    render_classification_card(classification)
    st.divider()
    render_false_positive_warnings(st.session_state.false_positive_flags)
else:
    st.info("No classification yet. Run a simulation from the main AtmoTwin page.")

