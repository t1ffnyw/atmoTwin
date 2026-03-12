import streamlit as st

from config import GASES, SCENARIO_PRESETS


def init_state() -> None:
    """Initialize all session state keys with defaults. Call once in app.py."""
    defaults = {
        # planet parameters
        "star_type": "G dwarf / Sun (5800 K)",
        "orbital_distance_au": 1.0,
        "surface_temp_k": 288,
        "surface_pressure_bar": 1.0,
        "gases": {g: info["default"] for g, info in GASES.items()},
        # pipeline results (populated after simulation)
        "spectrum": None,  # dict: {"wavelength": [...], "depth": [...]}
        "classification": None,  # dict: {"label": str, "confidence": float, ...}
        "false_positive_flags": [],
        # explorer: saved scenarios for comparison
        "saved_scenarios": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_planet_params() -> dict:
    """Gather current planet config from session state."""
    return {
        "star_type": st.session_state.star_type,
        "orbital_distance_au": st.session_state.orbital_distance_au,
        "surface_temp_k": st.session_state.surface_temp_k,
        "surface_pressure_bar": st.session_state.surface_pressure_bar,
        "gases": dict(st.session_state.gases),
    }


def load_preset(name: str) -> None:
    """Overwrite session state with a named scenario preset."""
    preset = SCENARIO_PRESETS[name]
    st.session_state.star_type = preset["star"]
    st.session_state.orbital_distance_au = preset["orbital_distance_au"]
    st.session_state.surface_temp_k = preset["surface_temp_k"]
    st.session_state.surface_pressure_bar = preset["surface_pressure_bar"]
    st.session_state.gases = dict(preset["gases"])

    # Sync gas slider widget keys so the UI updates immediately after rerun.
    # Streamlit widgets keep their own state by `key`, which can override `value=...`.
    for gas_id, gas_value in st.session_state.gases.items():
        st.session_state[f"input_{gas_id}"] = float(gas_value)

    # clear stale results
    st.session_state.spectrum = None
    st.session_state.classification = None
    st.session_state.false_positive_flags = []

