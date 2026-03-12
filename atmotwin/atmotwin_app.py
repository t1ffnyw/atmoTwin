import tempfile
from pathlib import Path

import streamlit as st

from plot_spectrum import (
    call_psg_api,
    modify_atmosphere,
    plot_spectrum,
    calculate_molecule_contributions,
    plot_molecule_contributions,
)


st.set_page_config(page_title="AtmoTwin", page_icon="🌍", layout="wide")


CONFIG_PATH_DEFAULT = "atmotwin/modern_earth_LIFE_cfg.txt"

# Preset scenarios (ppmv)
PRESETS = {
    "Modern Earth": {"O2": 210000.0, "CH4": 1.8, "CO2": 400.0, "CO": 0.1, "H2O": 10000.0, "N2O": 0.32, "O3": 0.1},
    "Lifeless Earth": {"O2": 0.0, "CH4": 0.001, "CO2": 400.0, "CO": 0.1, "H2O": 10000.0, "N2O": 0.32, "O3": 0.0},
    "Archean Earth": {"O2": 0.0, "CH4": 1000.0, "CO2": 10000.0, "CO": 0.1, "H2O": 10000.0, "N2O": 0.1, "O3": 0.0},
    "Ocean Loss (False Positive)": {
        "O2": 200000.0,
        "CH4": 0.001,
        "CO2": 400.0,
        "CO": 500.0,
        "H2O": 1000.0,
        "N2O": 0.1,
        "O3": 0.1,
    },
}

SESSION_KEYS = {
    "O2": "o2",
    "CH4": "ch4",
    "CO2": "co2",
    "CO": "co",
    "H2O": "h2o",
    "N2O": "n2o",
    "O3": "o3",
}


def ensure_default_state():
    """Initialize session state sliders from the Modern Earth preset."""
    defaults = PRESETS["Modern Earth"]
    for gas_key, value in defaults.items():
        sess_key = SESSION_KEYS[gas_key]
        if sess_key not in st.session_state:
            st.session_state[sess_key] = float(value)


def main():
    # Sidebar: inputs
    with st.sidebar:
        st.title("🌍 AtmoTwin: Exoplanet Atmosphere Builder")

        ensure_default_state()

        st.markdown("#### Preset Atmospheres")
        col1, col2, col3, col4 = st.columns(4)
        preset_clicked = None
        with col1:
            if st.button("Modern Earth"):
                preset_clicked = "Modern Earth"
        with col2:
            if st.button("Lifeless Earth"):
                preset_clicked = "Lifeless Earth"
        with col3:
            if st.button("Archean Earth"):
                preset_clicked = "Archean Earth"
        with col4:
            if st.button("Ocean Loss (False Positive)"):
                preset_clicked = "Ocean Loss (False Positive)"

        # Apply preset if clicked
        if preset_clicked:
            preset_vals = PRESETS[preset_clicked]
            for gas_key, val in preset_vals.items():
                sess_key = SESSION_KEYS[gas_key]
                st.session_state[sess_key] = float(val)

        st.markdown("#### Atmospheric Composition (ppmv)")

        o2 = st.slider("O₂", 0.0, 210000.0, float(st.session_state["o2"]), step=1000.0)
        ch4 = st.slider("CH₄", 0.001, 1000.0, float(st.session_state["ch4"]))
        co2 = st.slider("CO₂", 100.0, 10000.0, float(st.session_state["co2"]), step=10.0)
        co = st.slider("CO", 0.0, 1000.0, float(st.session_state["co"]), step=0.1)
        h2o = st.slider("H₂O", 1000.0, 30000.0, float(st.session_state["h2o"]), step=500.0)
        n2o = st.slider("N₂O", 0.0, 10.0, float(st.session_state["n2o"]), step=0.01)
        o3 = st.slider("O₃", 0.0, 10.0, float(st.session_state["o3"]), step=0.01)

        # Store back to session state
        st.session_state["o2"] = o2
        st.session_state["ch4"] = ch4
        st.session_state["co2"] = co2
        st.session_state["co"] = co
        st.session_state["h2o"] = h2o
        st.session_state["n2o"] = n2o
        st.session_state["o3"] = o3

        generate = st.button("Generate Spectrum")

    st.title("AtmoTwin: Build and Explore Exoplanet Atmospheres")

    config_path = Path(CONFIG_PATH_DEFAULT)
    if not config_path.exists():
        st.error(f"Base PSG configuration file not found: {config_path}")
        return

    if generate:
        with st.spinner("Contacting PSG and generating spectrum..."):
            # Build modified configuration
            cfg_str = modify_atmosphere(
                str(config_path),
                o2_ppmv=o2,
                ch4_ppmv=ch4,
                co2_ppmv=co2,
                o3_ppmv=o3,
                n2o_ppmv=n2o,
                co_ppmv=co,
                h2o_ppmv=h2o,
                n2_ppmv=0.0,
            )

            # Write to a temporary file for call_psg_api
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
                tmp.write(cfg_str)
                tmp_path = tmp.name

            try:
                wavelength, flux = call_psg_api(tmp_path, output_type="rad")
            except Exception as e:
                st.error(f"Error contacting PSG API: {e}")
                return
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        # Main spectrum panel
        fig, ax = plot_spectrum(wavelength, flux, title="Thermal Emission Spectrum")
        st.pyplot(fig)

        # Molecule contribution panel
        try:
            contrib_w, contrib_dict = calculate_molecule_contributions(
                str(config_path),
                o2_ppmv=o2,
                ch4_ppmv=ch4,
                co2_ppmv=co2,
                o3_ppmv=o3,
                n2o_ppmv=n2o,
                co_ppmv=co,
                h2o_ppmv=h2o,
                n2_ppmv=0.0,
                output_type="rad",
            )

            st.subheader("Molecular contributions across wavelength")
            st.markdown(
                "Each colored band shows how much a single molecule contributes "
                "to the overall spectrum at each wavelength. Overlapping regions "
                "indicate wavelengths where multiple molecules shape the signal."
            )
            fig_contrib, _ = plot_molecule_contributions(contrib_w, contrib_dict, stacked=True)
            st.pyplot(fig_contrib)
        except Exception as e:
            st.warning(f"Could not compute per-molecule contributions: {e}")

        # Atmospheric analysis
        st.subheader("🔬 Atmospheric Analysis")

        # Disequilibrium detection
        if o2 > 1000.0 and ch4 > 1.0:
            st.success(
                "✅ Strong chemical disequilibrium detected: O₂ + CH₄ coexistence suggests active biological processes"
            )
        elif co2 > 1000.0 and ch4 > 100.0 and co < 10.0:
            st.success("✅ CO₂ + CH₄ without CO — possible methanogenic biosphere")
        else:
            st.info("ℹ️ Atmosphere appears consistent with chemical equilibrium")

        # False positive warnings
        if o2 > 10000.0 and co > 100.0:
            st.warning(
                "⚠️ High CO with O₂ detected — possible abiotic oxygen from CO₂ photolysis or ocean loss"
            )
        if o2 > 10000.0 and h2o < 2000.0:
            st.warning("⚠️ High O₂ with low H₂O — possible ocean loss scenario")

        # Biosignature summary as metrics
        st.markdown("**Biosignature Summary (ppmv)**")

        row1 = st.columns(4)
        with row1[0]:
            st.metric("O₂", f"{o2:.3g}")
        with row1[1]:
            st.metric("CH₄", f"{ch4:.3g}")
        with row1[2]:
            st.metric("CO₂", f"{co2:.3g}")
        with row1[3]:
            st.metric("H₂O", f"{h2o:.3g}")

        row2 = st.columns(4)
        with row2[0]:
            st.metric("CO", f"{co:.3g}")
        with row2[1]:
            st.metric("N₂O", f"{n2o:.3g}")
        with row2[2]:
            st.metric("O₃", f"{o3:.3g}")


if __name__ == "__main__":
    main()

