import sys
from pathlib import Path

import streamlit as st

_dashboard_root = Path(__file__).resolve().parent.parent
_project_root = _dashboard_root.parent

for _p in (_dashboard_root, _project_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from state import init_state
from ui import configure_page
from components.result_cards import render_upload_classification


configure_page()
init_state()

st.title("Classifier")
st.caption(
    "Runs the AtmoTwin 4-class Random Forest on the latest simulated spectrum "
    "(or on uploaded spectra from the Spectrum page)."
)

st.divider()

result = st.session_state.get("classification")
if not result:
    result = st.session_state.get("_upload_classification")

if not result:
    st.info("No classification yet. Run **Simulate** on the Builder page, or upload a CSV on the Spectrum page.")
    st.stop()

render_upload_classification(result)