import streamlit as st

from config import STAR_PRESETS, GASES


def render_star_selector() -> None:
    st.selectbox(
        "Host Star",
        options=list(STAR_PRESETS.keys()),
        key="star_type",
    )


def render_orbital_params() -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input(
            "Orbital Distance (AU)",
            0.01,
            10.0,
            key="orbital_distance_au",
            step=0.05,
            format="%.2f",
        )
    with col2:
        st.number_input(
            "Surface Temp (K)",
            150,
            600,
            key="surface_temp_k",
            step=5,
        )
    with col3:
        st.number_input(
            "Surface Pressure (bar)",
            0.01,
            100.0,
            key="surface_pressure_bar",
            step=0.1,
            format="%.2f",
        )


def render_gas_sliders() -> None:
    """Render a slider for each gas (log10 mixing ratio)."""
    st.markdown("**Atmospheric Composition** *(log₁₀ mixing ratio)*")
    gases = st.session_state.gases
    cols = st.columns(2)
    for i, (gas_id, info) in enumerate(GASES.items()):
        with cols[i % 2]:
            gases[gas_id] = st.slider(
                info["label"],
                min_value=info["range"][0],
                max_value=info["range"][1],
                value=gases[gas_id],
                step=0.1,
                format="%.1f",
                key=f"slider_{gas_id}",
            )
    st.session_state.gases = gases

