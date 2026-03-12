# atmotwin_ml/data_loader.py

import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CLASS_NAMES = ['modern_earth', 'archean_earth', 'lifeless', 'false_positive']

CLASS_DESCRIPTIONS = {
    0: 'Inhabited (Modern) - Oxygenic photosynthesis',
    1: 'Inhabited (Archean) - Methanogenic biosphere', 
    2: 'Lifeless - Chemical equilibrium',
    3: 'False Positive - Abiotic O2 (waterworld)'
}

# Key diagnostic wavelengths (µm) from literature
DIAGNOSTIC_WAVELENGTHS = {
    'o3': 9.6,      # Ozone - biosignature (from O2)
    'ch4': 7.7,     # Methane - biosignature
    'n2o': 7.8,     # Nitrous oxide - biogenic only
    'co': 4.67,     # Carbon monoxide - anti-biosignature
    'h2o': 6.3,     # Water - habitability
    'co2': 15.0,    # Carbon dioxide - atmosphere bulk
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(npz_path='psg_dataset/atmotwin_training_data.npz'):
    """Load training data from npz file."""
    
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'wavelengths': data['wavelengths'],
        'spectra': data['spectra'],
        'params': data['params'],
        'labels': data['labels'],
        'class_names': list(data['class_names']),
        'gas_names': list(data['gas_names'])
    }

def load_data_csv(csv_path='psg_dataset/training_data.csv'):
    """Load training data from CSV file."""
    
    df = pd.read_csv(csv_path)
    
    # Separate columns
    class_names = df['class_name'].values
    labels = df['label'].values
    
    # Gas columns (between label and wavelength columns)
    gas_cols = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2']
    params = df[gas_cols].values
    
    # Wavelength columns (start with 'wl_')
    wl_cols = [c for c in df.columns if c.startswith('wl_')]
    wavelengths = np.array([float(c.replace('wl_', '')) for c in wl_cols])
    spectra = df[wl_cols].values
    
    return {
        'wavelengths': wavelengths,
        'spectra': spectra,
        'params': params,
        'labels': labels,
        'class_names_per_sample': class_names,
        'gas_names': gas_cols
    }

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def get_flux_at_wavelength(spectrum, wavelengths, target_wl, width=0.2):
    """
    Extract flux value near a target wavelength.
    
    Args:
        spectrum: 1D array of flux values
        wavelengths: 1D array of wavelength values
        target_wl: Target wavelength in µm
        width: Wavelength window for averaging (µm)
    
    Returns:
        Mean flux in the wavelength window
    """
    mask = np.abs(wavelengths - target_wl) < width
    
    if mask.sum() == 0:
        # No points in window - use nearest
        idx = np.argmin(np.abs(wavelengths - target_wl))
        return spectrum[idx]
    
    return spectrum[mask].mean()

def extract_band_depths(spectrum, wavelengths):
    """
    Extract absorption depths at key diagnostic wavelengths.
    
    Returns dict of band depths for each molecule.
    """
    features = {}
    
    for molecule, wl in DIAGNOSTIC_WAVELENGTHS.items():
        features[f'{molecule}_depth'] = get_flux_at_wavelength(
            spectrum, wavelengths, wl
        )
    
    return features

def extract_disequilibrium_features(band_depths):
    """
    Compute chemical disequilibrium indicators.
    
    Scientific basis: O3 + CH4 coexistence indicates biology
    (Quanz et al. 2019, Meadows et al. 2023)
    """
    features = {}
    
    # O3/CH4 ratio - high when both present (modern Earth biosignature)
    # Adding small epsilon to avoid division by zero
    eps = 1e-10
    
    features['o3_ch4_ratio'] = (
        band_depths['o3_depth'] / (band_depths['ch4_depth'] + eps)
    )
    
    # CO/O3 ratio - high indicates false positive (abiotic O2)
    # Waterworlds have elevated CO with O2/O3
    features['co_o3_ratio'] = (
        band_depths['co_depth'] / (band_depths['o3_depth'] + eps)
    )
    
    # CH4/CO2 ratio - distinguishes Archean (high CH4) from lifeless
    features['ch4_co2_ratio'] = (
        band_depths['ch4_depth'] / (band_depths['co2_depth'] + eps)
    )
    
    # N2O presence indicator - N2O is almost exclusively biogenic
    features['n2o_depth'] = band_depths['n2o_depth']
    
    return features

def engineer_features(spectra, wavelengths, include_raw=True):
    """
    Create full feature matrix with engineered features.
    
    Args:
        spectra: 2D array (n_samples, n_wavelengths)
        wavelengths: 1D array of wavelength values
        include_raw: Whether to include raw spectrum values
    
    Returns:
        X: Feature matrix
        feature_names: List of feature names
    """
    n_samples = spectra.shape[0]
    
    engineered_features = []
    feature_names_eng = []
    
    for i in range(n_samples):
        # Extract band depths
        band_depths = extract_band_depths(spectra[i], wavelengths)
        
        # Compute disequilibrium features
        diseq_features = extract_disequilibrium_features(band_depths)
        
        # Combine all engineered features
        all_eng = {**band_depths, **diseq_features}
        engineered_features.append(list(all_eng.values()))
        
        if i == 0:
            feature_names_eng = list(all_eng.keys())
    
    engineered_array = np.array(engineered_features)
    
    if include_raw:
        # Combine raw spectrum + engineered features
        X = np.column_stack([spectra, engineered_array])
        feature_names = [f'wl_{w:.2f}' for w in wavelengths] + feature_names_eng
    else:
        X = engineered_array
        feature_names = feature_names_eng
    
    return X, feature_names

# =============================================================================
# DATA PREPARATION PIPELINE
# =============================================================================

def prepare_dataset(data_path='psg_dataset/atmotwin_training_data.npz', 
                    include_raw_spectrum=True):
    """
    Full pipeline: load data and engineer features.
    
    Returns:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        metadata: Dict with wavelengths, class info, etc.
    """
    # Load data
    if str(data_path).endswith('.npz'):
        data = load_data(data_path)
    else:
        data = load_data_csv(data_path)
    
    # Engineer features
    X, feature_names = engineer_features(
        data['spectra'], 
        data['wavelengths'],
        include_raw=include_raw_spectrum
    )
    
    y = data['labels']
    
    metadata = {
        'wavelengths': data['wavelengths'],
        'class_names': CLASS_NAMES,
        'class_descriptions': CLASS_DESCRIPTIONS,
        'gas_names': data['gas_names'],
        'n_samples': len(y),
        'n_features': X.shape[1],
        'n_raw_features': len(data['wavelengths']),
        'n_engineered_features': len(feature_names) - len(data['wavelengths']) if include_raw_spectrum else len(feature_names)
    }
    
    return X, y, feature_names, metadata

# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    print("Loading and preparing dataset...")
    print("=" * 60)
    
    X, y, feature_names, metadata = prepare_dataset()
    
    print(f"Samples: {metadata['n_samples']}")
    print(f"Total features: {metadata['n_features']}")
    print(f"  - Raw wavelength bins: {metadata['n_raw_features']}")
    print(f"  - Engineered features: {metadata['n_engineered_features']}")
    
    print(f"\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y == i)
        print(f"  {i}: {name} - {count} samples")
    
    print(f"\nEngineered features:")
    eng_features = feature_names[-metadata['n_engineered_features']:]
    for f in eng_features:
        print(f"  - {f}")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    print("\n✅ Data loading module ready!")