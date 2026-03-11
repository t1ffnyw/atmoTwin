import numpy as np

# --- Star presets (T_eff in K, luminosity relative to Sun) ---
STAR_PRESETS = {
    "M dwarf (3300 K)": {"t_eff": 3300, "luminosity": 0.04},
    "K dwarf (4500 K)": {"t_eff": 4500, "luminosity": 0.3},
    "G dwarf / Sun (5800 K)": {"t_eff": 5800, "luminosity": 1.0},
    "F dwarf (6500 K)": {"t_eff": 6500, "luminosity": 2.5},
}

# --- Atmospheric gas options with plausible ranges (log10 mixing ratio) ---
GASES = {
    "N2": {"label": "N₂", "default": -0.1, "range": (-2.0, 0.0)},
    "O2": {"label": "O₂", "default": -0.68, "range": (-8.0, 0.0)},
    "CO2": {"label": "CO₂", "default": -3.5, "range": (-8.0, -0.3)},
    "CH4": {"label": "CH₄", "default": -5.2, "range": (-8.0, -1.0)},
    "H2O": {"label": "H₂O", "default": -2.5, "range": (-8.0, -0.5)},
    "CO": {"label": "CO", "default": -7.0, "range": (-8.0, -2.0)},
    "N2O": {"label": "N₂O", "default": -6.5, "range": (-8.0, -3.0)},
    "O3": {"label": "O₃", "default": -5.8, "range": (-8.0, -3.0)},
}

# --- Biosignature spectral bands (wavelength in μm) ---
BIOSIG_BANDS = {
    "O₂ (A-band)": {"center": 0.76, "width": 0.02, "gas": "O2", "color": "#3b82f6"},
    "O₂ (1.27 μm)": {"center": 1.27, "width": 0.03, "gas": "O2", "color": "#60a5fa"},
    "CH₄ (1.6 μm)": {"center": 1.65, "width": 0.10, "gas": "CH4", "color": "#f59e0b"},
    "CH₄ (3.3 μm)": {"center": 3.3, "width": 0.20, "gas": "CH4", "color": "#fbbf24"},
    "CO₂ (4.3 μm)": {"center": 4.3, "width": 0.30, "gas": "CO2", "color": "#ef4444"},
    "O₃ (9.6 μm)": {"center": 9.6, "width": 0.50, "gas": "O3", "color": "#8b5cf6"},
    "H₂O (6.3 μm)": {"center": 6.3, "width": 0.40, "gas": "H2O", "color": "#06b6d4"},
    "CO₂ (15 μm)": {"center": 15.0, "width": 1.00, "gas": "CO2", "color": "#f87171"},
    "N₂O (4.5 μm)": {"center": 4.5, "width": 0.20, "gas": "N2O", "color": "#a78bfa"},
}

# --- Scenario presets from Meadows et al. 2023 ---
SCENARIO_PRESETS = {
    "Modern Earth": {
        "star": "G dwarf / Sun (5800 K)",
        "orbital_distance_au": 1.0,
        "surface_temp_k": 288,
        "surface_pressure_bar": 1.0,
        "gases": {
            "N2": -0.1,
            "O2": -0.68,
            "CO2": -3.5,
            "CH4": -5.2,
            "H2O": -2.5,
            "CO": -7.0,
            "N2O": -6.5,
            "O3": -5.8,
        },
        "category": "biological",
    },
    "Archean Earth": {
        "star": "G dwarf / Sun (5800 K)",
        "orbital_distance_au": 1.0,
        "surface_temp_k": 290,
        "surface_pressure_bar": 1.0,
        "gases": {
            "N2": -0.1,
            "O2": -6.0,
            "CO2": -1.5,
            "CH4": -3.0,
            "H2O": -2.5,
            "CO": -5.0,
            "N2O": -8.0,
            "O3": -8.0,
        },
        "category": "biological",
    },
    "Volcanic / Prebiotic": {
        "star": "M dwarf (3300 K)",
        "orbital_distance_au": 0.15,
        "surface_temp_k": 280,
        "surface_pressure_bar": 1.0,
        "gases": {
            "N2": -0.1,
            "O2": -7.0,
            "CO2": -1.0,
            "CH4": -5.5,
            "H2O": -2.0,
            "CO": -3.5,
            "N2O": -8.0,
            "O3": -8.0,
        },
        "category": "false_positive",
    },
    "Ocean Loss (O₂ buildup)": {
        "star": "M dwarf (3300 K)",
        "orbital_distance_au": 0.15,
        "surface_temp_k": 350,
        "surface_pressure_bar": 1.0,
        "gases": {
            "N2": -0.3,
            "O2": -0.3,
            "CO2": -2.0,
            "CH4": -8.0,
            "H2O": -3.5,
            "CO": -4.0,
            "N2O": -8.0,
            "O3": -4.5,
        },
        "category": "false_positive",
    },
}

