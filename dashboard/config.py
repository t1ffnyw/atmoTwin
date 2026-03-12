import numpy as np

# --- Star presets (T_eff in K, luminosity relative to Sun) ---
STAR_PRESETS = {
    "M dwarf (3300 K)": {"t_eff": 3300, "luminosity": 0.04},
    "K dwarf (4500 K)": {"t_eff": 4500, "luminosity": 0.3},
    "G dwarf / Sun (5800 K)": {"t_eff": 5800, "luminosity": 1.0},
    "F dwarf (6500 K)": {"t_eff": 6500, "luminosity": 2.5},
}

# --- Atmospheric gas options (ppmv), ranges from atmotwin_app.py ---
GASES = {
    "O2": {"label": "O₂", "default": 210000.0, "range": (0.0, 600000.0), "step": 1000.0},
    "CH4": {"label": "CH₄", "default": 1.8, "range": (0.001, 1000.0), "step": 0.1},
    "CO2": {"label": "CO₂", "default": 400.0, "range": (0.0, 200000.0), "step": 10.0},
    "CO": {"label": "CO", "default": 0.1, "range": (0.0, 5000.0), "step": 0.1},
    "H2O": {"label": "H₂O", "default": 10000.0, "range": (0.0, 30000.0), "step": 500.0},
    "N2O": {"label": "N₂O", "default": 0.32, "range": (0.0, 10.0), "step": 0.01},
    "O3": {"label": "O₃", "default": 0.1, "range": (0.0, 50.0), "step": 0.01},
    "N2": {"label": "N₂", "default": 780000.0, "range": (0.0, 800000.0), "step": 10000.0},
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
    "CO (4.67 μm)": {"center": 4.67, "width": 0.15, "gas": "CO", "color": "#1abc9c"},
}

# --- Scenario presets (gases in ppmv) ---
SCENARIO_PRESETS = {
    "Modern Earth": {
        "star": "G dwarf / Sun (5800 K)",
        "orbital_distance_au": 1.0,
        "surface_temp_k": 288,
        "surface_pressure_bar": 1.0,
        "gases": {
            "O2": 210000.0,
            "CH4": 1.8,
            "CO2": 400.0,
            "CO": 0.1,
            "H2O": 10000.0,
            "N2O": 0.32,
            "O3": 0.1,
            "N2": 780000.0,
        },
        "category": "biological",
    },
    "Lifeless Earth": {
        "star": "G dwarf / Sun (5800 K)",
        "orbital_distance_au": 1.0,
        "surface_temp_k": 288,
        "surface_pressure_bar": 1.0,
        "gases": {
            "O2": 0.0,
            "CH4": 0.001,
            "CO2": 400.0,
            "CO": 0.1,
            "H2O": 10000.0,
            "N2O": 0.0,
            "O3": 0.0,
            "N2": 780000.0,
        },
        "category": "abiotic",
    },
    "Archean Earth": {
        "star": "G dwarf / Sun (5800 K)",
        "orbital_distance_au": 1.0,
        "surface_temp_k": 290,
        "surface_pressure_bar": 1.0,
        "gases": {
            "O2": 0.0,
            "CH4": 1000.0,
            "CO2": 10000.0,
            "CO": 0.1,
            "H2O": 10000.0,
            "N2O": 0.0,
            "O3": 0.0,
            "N2": 780000.0,
        },
        "category": "biological",
    },
    "Volcanic / Prebiotic": {
        "star": "M dwarf (3300 K)",
        "orbital_distance_au": 0.15,
        "surface_temp_k": 280,
        "surface_pressure_bar": 1.0,
        "gases": {
            "O2": 0.1,
            "CH4": 316.0,
            "CO2": 100000.0,
            "CO": 3162.0,
            "H2O": 10000.0,
            "N2O": 0.01,
            "O3": 0.01,
            "N2": 780000.0,
        },
        "category": "false_positive",
    },
    "Ocean Loss (O₂ buildup)": {
        "star": "M dwarf (3300 K)",
        "orbital_distance_au": 0.15,
        "surface_temp_k": 350,
        "surface_pressure_bar": 1.0,
        "gases": {
            "O2": 501187.0,
            "CH4": 0.01,
            "CO2": 10000.0,
            "CO": 100.0,
            "H2O": 3162.0,
            "N2O": 0.01,
            "O3": 31.6,
            "N2": 501187.0,
        },
        "category": "false_positive",
    },
}
