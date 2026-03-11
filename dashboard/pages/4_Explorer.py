import copy

import streamlit as st

from config import SCENARIO_PRESETS
from state import init_state, get_planet_params, load_preset
from ui import configure_page


configure_page()
init_state()

st.title("Scenario Explorer")

st.markdown(
    "Save different atmospheric configurations and compare them side by side. "
    "Use presets as starting points, adjust, then save snapshots."
)

st.divider()

col_actions, col_list = st.columns([1, 2])

with col_actions:
    if st.button("Save current scenario"):
        current = get_planet_params()
        st.session_state.saved_scenarios.append(current)
        st.success("Saved current scenario.")

    if st.session_state.saved_scenarios and st.button("Clear saved scenarios"):
        st.session_state.saved_scenarios = []
        st.info("Cleared saved scenarios.")

with col_list:
    if not st.session_state.saved_scenarios:
        st.info("No saved scenarios yet. Save from the main app, then compare here.")
    else:
        st.subheader("Saved scenarios")
        for i, scenario in enumerate(st.session_state.saved_scenarios):
            with st.expander(f"Scenario {i + 1}"):
                st.json(scenario)

st.divider()

st.subheader("Preset quick load")
preset = st.selectbox("Load preset into current state", ["(none)"] + list(SCENARIO_PRESETS.keys()))
if preset != "(none)":
    load_preset(preset)
    st.success(f"Loaded preset: {preset}")

