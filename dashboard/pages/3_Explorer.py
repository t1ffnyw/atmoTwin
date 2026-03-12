import sys
from pathlib import Path

import streamlit as st

_dashboard_root = Path(__file__).resolve().parent.parent
if str(_dashboard_root) not in sys.path:
    sys.path.insert(0, str(_dashboard_root))

from state import init_state, get_planet_params
from components.spectrum_plot import make_comparison_figure
from psg.service import generate_comparison_spectra
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
        n = len(st.session_state.saved_scenarios) + 1
        current["name"] = f"Scenario {n}"
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
            if "name" not in scenario:
                scenario["name"] = f"Scenario {i + 1}"
            with st.expander(scenario["name"]):
                new_name = st.text_input(
                    "Name",
                    value=scenario["name"],
                    key=f"rename_{i}",
                )
                if new_name != scenario["name"]:
                    scenario["name"] = new_name
                    st.rerun()
                st.json({k: v for k, v in scenario.items() if k != "name"})

# ── Spectrum Comparison ──────────────────────────────────────────────
st.divider()
st.subheader("Compare Two Scenarios")

saved = st.session_state.saved_scenarios
scenario_labels = [s.get("name", f"Scenario {i + 1}") for i, s in enumerate(saved)]

if len(saved) < 2:
    st.info("Save at least **2 scenarios** to unlock the comparison chart.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        idx_a = st.selectbox("First scenario", range(len(saved)), format_func=lambda i: scenario_labels[i], key="cmp_a")
    with col_b:
        idx_b = st.selectbox("Second scenario", range(len(saved)), format_func=lambda i: scenario_labels[i], index=min(1, len(saved) - 1), key="cmp_b")

    if idx_a == idx_b:
        st.warning("Select two different scenarios to compare.")
    else:
        compare_clicked = st.button("Compare Spectra", type="primary")

        if compare_clicked:
            with st.spinner("Generating spectra for both scenarios via PSG..."):
                try:
                    wavelength, fluxes = generate_comparison_spectra(
                        [saved[idx_a], saved[idx_b]],
                    )
                    st.session_state.comparison_result = {
                        "wavelength": wavelength,
                        "fluxes": fluxes,
                        "labels": [scenario_labels[idx_a], scenario_labels[idx_b]],
                    }
                except FileNotFoundError as e:
                    st.error(f"Configuration error: {e}")
                except Exception as e:
                    st.error(f"PSG error: {e}")

        if st.session_state.get("comparison_result"):
            cmp = st.session_state.comparison_result
            fig = make_comparison_figure(cmp["wavelength"], cmp["fluxes"], cmp["labels"])
            st.plotly_chart(fig, width="stretch")
