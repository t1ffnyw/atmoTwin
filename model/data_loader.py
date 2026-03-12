"""
AtmoTwin Data Loader and Feature Engineering

Provides feature extraction for both training and inference.
"""

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

CLASS_NAMES = ["modern_earth", "archean_earth", "lifeless", "false_positive"]

CLASS_DESCRIPTIONS = {
    0: "Modern Earth - Oxygenic photosynthesis biosphere",
    1: "Archean Earth - Methanogenic biosphere (anoxic)",
    2: "Lifeless - Chemical equilibrium, no biology",
    3: "False Positive - Abiotic O₂ (e.g., waterworld)",
}

# Molecule absorption bands (µm)
MOLECULE_BANDS = {
    "O3": {"peak": 9.6, "range": (9.0, 10.5), "color": "#2ecc71"},
    "CH4": {"peak": 7.7, "range": (7.2, 8.2), "color": "#9b59b6"},
    "N2O": {"peak": 7.8, "range": (7.6, 8.0), "color": "#e74c3c"},
    "CO": {"peak": 4.67, "range": (4.4, 4.9), "color": "#1abc9c"},
    "H2O": {"peak": 6.3, "range": (5.5, 7.5), "color": "#3498db"},
    "CO2": {"peak": 15.0, "range": (14.0, 16.5), "color": "#e67e22"},
}

# Key diagnostic wavelengths (µm)
DIAGNOSTIC_WAVELENGTHS = {
    "o3": 9.6,
    "ch4": 7.7,
    "n2o": 7.8,
    "co": 4.67,
    "h2o": 6.3,
    "co2": 15.0,
}


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def find_nearest_index(wavelengths: np.ndarray, target: float) -> int:
    """Find index of wavelength nearest to target."""
    return int(np.argmin(np.abs(wavelengths - target)))


def get_continuum_estimate(
    spectrum: np.ndarray, wavelengths: np.ndarray, center: float, width: float = 1.0
) -> float:
    """
    Estimate continuum level near an absorption band.
    Uses average of flux values outside the band.
    """
    left_idx = find_nearest_index(wavelengths, center - width)
    right_idx = find_nearest_index(wavelengths, center + width)
    
    # Get values at band edges
    left_val = spectrum[max(0, left_idx - 2) : left_idx].mean() if left_idx > 2 else spectrum[0]
    right_val = spectrum[right_idx : min(len(spectrum), right_idx + 3)].mean() if right_idx < len(spectrum) - 3 else spectrum[-1]
    
    return (left_val + right_val) / 2


def extract_band_depths(spectrum: np.ndarray, wavelengths: np.ndarray) -> dict:
    """
    Extract absorption band depths for key molecules.

    Band depth = (continuum - band_center) / continuum

    Args:
        spectrum: 1D flux array
        wavelengths: 1D wavelength array (µm)

    Returns:
        Dict of band depth values
    """
    band_depths = {}
    eps = 1e-10

    for mol, wl in DIAGNOSTIC_WAVELENGTHS.items():
        idx = find_nearest_index(wavelengths, wl)
        band_flux = spectrum[idx]
        continuum = get_continuum_estimate(spectrum, wavelengths, wl)
        
        depth = (continuum - band_flux) / (continuum + eps)
        band_depths[f"{mol}_depth"] = max(0, depth)  # Clip negative values

    return band_depths


def extract_disequilibrium_features(band_depths: dict) -> dict:
    """
    Compute disequilibrium ratio features from band depths.

    These ratios help distinguish biological from abiotic scenarios.
    """
    eps = 1e-10
    features = {}

    # O3/CH4 ratio - high in Modern Earth (O2 photolysis creates O3)
    features["o3_ch4_ratio"] = band_depths["o3_depth"] / (band_depths["ch4_depth"] + eps)

    # CO/O3 ratio - high CO with O3 indicates false positive
    features["co_o3_ratio"] = band_depths["co_depth"] / (band_depths["o3_depth"] + eps)

    # CH4/CO2 ratio - high in Archean (methanogenic biosphere)
    features["ch4_co2_ratio"] = band_depths["ch4_depth"] / (band_depths["co2_depth"] + eps)

    return features


def engineer_features(
    spectra: np.ndarray, wavelengths: np.ndarray, include_raw: bool = True
) -> tuple:
    """
    Create full feature matrix with engineered features.

    Args:
        spectra: 2D array (n_samples, n_wavelengths) or 1D array
        wavelengths: 1D array of wavelength values
        include_raw: Whether to include raw spectrum values

    Returns:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
    """
    # Handle 1D input
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)

    n_samples = spectra.shape[0]

    engineered_features = []
    feature_names_eng = []

    for i in range(n_samples):
        band_depths = extract_band_depths(spectra[i], wavelengths)
        diseq_features = extract_disequilibrium_features(band_depths)

        all_eng = {**band_depths, **diseq_features}
        engineered_features.append(list(all_eng.values()))

        if i == 0:
            feature_names_eng = list(all_eng.keys())

    engineered_array = np.array(engineered_features)

    if include_raw:
        X = np.column_stack([spectra, engineered_array])
        feature_names = [f"wl_{w:.2f}" for w in wavelengths] + feature_names_eng
    else:
        X = engineered_array
        feature_names = feature_names_eng

    return X, feature_names


# =============================================================================
# DATA LOADING (for training - not needed for inference)
# =============================================================================

def load_data(npz_path: str = "psg_dataset/atmotwin_training_data.npz") -> dict:
    """Load training data from npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "wavelengths": data["wavelengths"],
        "spectra": data["spectra"],
        "params": data["params"],
        "labels": data["labels"],
        "class_names": list(data["class_names"]),
        "gas_names": list(data["gas_names"]),
    }


def prepare_dataset(
    data_path: str = "psg_dataset/atmotwin_training_data.npz",
    include_raw_spectrum: bool = True,
) -> tuple:
    """
    Full pipeline: load data and engineer features.

    Returns:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        metadata: Dict with wavelengths, class info, etc.
    """
    data = load_data(data_path)

    X, feature_names = engineer_features(
        data["spectra"], data["wavelengths"], include_raw=include_raw_spectrum
    )

    y = data["labels"]

    metadata = {
        "wavelengths": data["wavelengths"],
        "class_names": data["class_names"],
        "gas_names": data["gas_names"],
        "n_samples": len(y),
        "n_features": X.shape[1],
    }

    return X, y, feature_names, metadata