"""
AtmoTwin ML Inference Module

Provides classification and explanation for exoplanet spectra.
Integrates with the Streamlit dashboard.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

from .data_loader import (
    engineer_features,
    extract_band_depths,
    extract_disequilibrium_features,
    MOLECULE_BANDS,
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
)

# =============================================================================
# PATHS
# =============================================================================

_MODEL_DIR = Path(__file__).resolve().parent

# =============================================================================
# CONSTANTS
# =============================================================================

_WL_MIN = 4.0    # µm — lower bound of training grid
_WL_MAX = 18.33  # µm — upper bound of training grid
_WL_TOLERANCE = 0.5  # µm — tolerance for user wavelength range

# =============================================================================
# MODEL LOADING
# =============================================================================

_MODEL_CACHE: Optional[tuple] = None
_TRAIN_SCALE: Optional[float] = None


def _load_training_scale() -> float:
    """
    Compute the mean flux value of the training spectra so we can rescale
    dashboard spectra (which may come from a different PSG server) to the
    same absolute level the RF model was trained on.
    """
    global _TRAIN_SCALE
    if _TRAIN_SCALE is not None:
        return _TRAIN_SCALE

    npz_path = _MODEL_DIR / "atmotwin_training_data.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        _TRAIN_SCALE = float(np.mean(data["spectra"]))
    else:
        _TRAIN_SCALE = 0.0
    return _TRAIN_SCALE


def load_model():
    """
    Load trained RF model, feature names, and metadata from disk.

    Returns:
        model: fitted RandomForestClassifier
        feature_names: list of feature name strings
        metadata: dict from model_metadata.json
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    model_path = _MODEL_DIR / "atmotwin_classifier_robust.joblib"
    if not model_path.exists():
        model_path = _MODEL_DIR / "atmotwin_classifier.joblib"
    
    model = joblib.load(model_path)

    with (_MODEL_DIR / "feature_names.json").open() as f:
        feature_names = json.load(f)

    with (_MODEL_DIR / "model_metadata.json").open() as f:
        metadata = json.load(f)

    _MODEL_CACHE = (model, feature_names, metadata)
    return _MODEL_CACHE


def clear_model_cache():
    """Clear the model cache (useful for testing)."""
    global _MODEL_CACHE, _TRAIN_SCALE
    _MODEL_CACHE = None
    _TRAIN_SCALE = None


# =============================================================================
# INTERPOLATION
# =============================================================================

def interpolate_to_training_grid(
    user_wavelengths: np.ndarray,
    user_flux: np.ndarray,
    training_wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Interpolate uploaded spectrum to the model's exact wavelength grid.

    Args:
        user_wavelengths: 1D array, sorted ascending (µm)
        user_flux: 1D array of flux values
        training_wavelengths: 1D array of target wavelength bins

    Returns:
        np.ndarray of interpolated flux values

    Raises:
        ValueError: if user wavelengths don't cover required range
    """
    if user_wavelengths[0] > _WL_MIN + _WL_TOLERANCE:
        raise ValueError(
            f"Spectrum must start at or before {_WL_MIN:.2f} µm. "
            f"Your data starts at {user_wavelengths[0]:.2f} µm."
        )
    if user_wavelengths[-1] < _WL_MAX - _WL_TOLERANCE:
        raise ValueError(
            f"Spectrum must extend to at least {_WL_MAX:.2f} µm. "
            f"Your data ends at {user_wavelengths[-1]:.2f} µm."
        )

    return np.interp(training_wavelengths, user_wavelengths, user_flux)


# =============================================================================
# MOLECULE EXPLANATION
# =============================================================================

def explain_prediction(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    predicted_class: str,
    band_depths: dict,
) -> list:
    """
    Explain which molecules contributed to this specific prediction.

    Args:
        spectrum: 1D flux array
        wavelengths: 1D wavelength array
        predicted_class: The predicted class name
        band_depths: Dict of band depth values

    Returns:
        List of contributing molecule dicts, sorted by importance
    """
    
    contributions = []
    
    for mol, info in MOLECULE_BANDS.items():
        depth = band_depths.get(f"{mol.lower()}_depth", 0)
        
        # Determine absorption strength
        if depth > 0.5:
            strength = "strong"
        elif depth > 0.2:
            strength = "moderate"
        elif depth > 0.05:
            strength = "weak"
        else:
            strength = "minimal"
        
        # Skip minimal contributions
        if strength == "minimal":
            continue
        
        # Get interpretation based on molecule and predicted class
        interpretation = get_molecule_interpretation(mol, predicted_class, depth)
        
        contributions.append({
            "molecule": mol,
            "wavelength": info["peak"],
            "absorption_depth": float(depth),
            "strength": strength,
            "interpretation": interpretation,
            "color": info["color"],
        })
    
    # Sort by absorption depth (strongest first)
    contributions.sort(key=lambda x: x["absorption_depth"], reverse=True)
    
    return contributions


_GAS_TO_DEPTH_FEATURE = {
    "O3": "o3_depth",
    "CH4": "ch4_depth",
    "N2O": "n2o_depth",
    "CO": "co_depth",
    "H2O": "h2o_depth",
    "CO2": "co2_depth",
}


def get_molecule_explanations(
    model,
    feature_names: list,
    training_wavelengths: np.ndarray,
    biosig_bands: dict,
    band_depths: dict,
    predicted_class: str,
) -> list:
    """
    Merge RF feature importances with measured band depths to produce
    a per-molecule explanation list.

    For each band in *biosig_bands* whose center falls within the model's
    wavelength range, the function:
      1. Averages the RF feature importances for wavelength bins inside the band.
      2. Includes importance of the matching engineered depth feature, if any.
      3. Combines this with the spectrum-specific absorption depth.
      4. Flags molecules as "significant" when their combined score exceeds
         1.5x the mean across all in-range bands.

    Returns a list of dicts sorted by descending combined importance.
    """
    importances = model.feature_importances_

    wl_min = training_wavelengths[0]
    wl_max = training_wavelengths[-1]

    wl_feature_wls = []
    wl_feature_idxs = []
    for i, name in enumerate(feature_names):
        if name.startswith("wl_"):
            wl_feature_wls.append(float(name[3:]))
            wl_feature_idxs.append(i)
    wl_feature_wls = np.array(wl_feature_wls)
    wl_feature_idxs = np.array(wl_feature_idxs)

    eng_feature_map = {name: i for i, name in enumerate(feature_names) if not name.startswith("wl_")}

    raw_entries = []
    for band_label, band_info in biosig_bands.items():
        center = band_info["center"]
        width = band_info["width"]
        gas = band_info.get("gas", "")
        color = band_info.get("color", "#94a3b8")

        if center + width < wl_min or center - width > wl_max:
            continue

        lo, hi = center - width, center + width
        mask = (wl_feature_wls >= lo) & (wl_feature_wls <= hi)

        band_importances = []
        if mask.any():
            band_importances.extend(importances[wl_feature_idxs[mask]].tolist())

        depth_feat = _GAS_TO_DEPTH_FEATURE.get(gas)
        if depth_feat and depth_feat in eng_feature_map:
            band_importances.append(importances[eng_feature_map[depth_feat]])

        avg_importance = float(np.mean(band_importances)) if band_importances else 0.0

        depth_val = band_depths.get(f"{gas.lower()}_depth", 0.0)

        if depth_val > 0.5:
            strength = "strong"
        elif depth_val > 0.2:
            strength = "moderate"
        elif depth_val > 0.05:
            strength = "weak"
        else:
            strength = "minimal"

        # Only report molecules actually present in the spectrum.
        if strength == "minimal":
            continue

        interpretation = get_molecule_interpretation(gas, predicted_class, depth_val)

        raw_entries.append({
            "molecule": band_label,
            "gas": gas,
            "band_center": center,
            "importance": avg_importance,
            "absorption_depth": float(depth_val),
            "strength": strength,
            "interpretation": interpretation,
            "color": color,
        })

    if not raw_entries:
        return []

    max_imp = max(e["importance"] for e in raw_entries) or 1.0
    for e in raw_entries:
        e["relative_importance"] = e["importance"] / max_imp

    combined_scores = []
    for e in raw_entries:
        combined = 0.4 * e["relative_importance"] + 0.6 * min(e["absorption_depth"] / 0.5, 1.0)
        e["combined_score"] = combined
        combined_scores.append(combined)

    mean_combined = float(np.mean(combined_scores)) if combined_scores else 0.0
    for e in raw_entries:
        e["significant"] = e["combined_score"] > 1.5 * mean_combined

    raw_entries.sort(key=lambda x: x["combined_score"], reverse=True)
    return raw_entries


def get_molecule_interpretation(molecule: str, predicted_class: str, depth: float) -> str:
    """Get human-readable interpretation for a molecule detection."""
    
    interpretations = {
        "O3": {
            "modern_earth": "Ozone from photolysis of biogenic O₂ - biosignature",
            "archean_earth": "Unexpected O₃ in anoxic atmosphere",
            "lifeless": "Minimal ozone consistent with lifeless planet",
            "false_positive": "Abiotic O₃ from photolysis - false positive indicator",
        },
        "CH4": {
            "modern_earth": "Biogenic methane in oxidizing atmosphere - requires replenishment",
            "archean_earth": "High CH₄ from methanogenic biosphere - Archean biosignature",
            "lifeless": "Low CH₄ consistent with chemical equilibrium",
            "false_positive": "CH₄ may be volcanic or abiotic in origin",
        },
        "N2O": {
            "modern_earth": "N₂O is almost exclusively biogenic - strong biosignature",
            "archean_earth": "Biogenic N₂O indicator",
            "lifeless": "N₂O should be absent on lifeless worlds",
            "false_positive": "N₂O presence challenges false positive hypothesis",
        },
        "CO": {
            "modern_earth": "Low CO consistent with active biosphere",
            "archean_earth": "Moderate CO in reducing atmosphere",
            "lifeless": "CO accumulation from photochemistry",
            "false_positive": "Elevated CO supports abiotic scenario",
        },
        "H2O": {
            "modern_earth": "Water vapor indicates habitable conditions",
            "archean_earth": "Water essential for life",
            "lifeless": "Water present but no life signatures",
            "false_positive": "Water loss may drive O₂ accumulation",
        },
        "CO2": {
            "modern_earth": "Moderate CO₂ with active carbon cycle",
            "archean_earth": "Higher CO₂ in early atmosphere",
            "lifeless": "CO₂-dominated atmosphere typical of lifeless worlds",
            "false_positive": "CO₂ photolysis can produce abiotic O₂",
        },
    }
    
    mol_interp = interpretations.get(molecule, {})
    return mol_interp.get(predicted_class, f"{molecule} detected at this wavelength")


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict(user_wavelengths, user_flux, biosig_bands=None):
    """
    Run the full inference pipeline on a spectrum.

    Args:
        user_wavelengths: 1D array of wavelength values (µm), sorted ascending
        user_flux: 1D array of flux/contrast values
        biosig_bands: Optional dict of biosignature band definitions (from
            dashboard config).  When provided the molecule explanation merges
            RF feature importances with measured band depths; otherwise falls
            back to the simpler band-depth-only explanation.

    Returns:
        dict with keys:
            class_names         list[str] - all class names
            probabilities       list[float] - probability for each class
            predicted_class     str - name of predicted class
            confidence          float - confidence of prediction
            is_inhabited        bool - True if modern_earth or archean_earth
            key_features        list[(str, float)] - top 5 important features
            diagnostics         dict[str, float] - band depths and ratios
            contributing_molecules  list[dict] - per-prediction explanation
            class_description   str - human-readable class description
    """
    model, feature_names, metadata = load_model()

    training_wavelengths = np.array(metadata["wavelengths"])

    interp_flux = interpolate_to_training_grid(
        np.asarray(user_wavelengths, dtype=float),
        np.asarray(user_flux, dtype=float),
        training_wavelengths,
    )

    # Rescale so absolute flux level matches the training data.  The model
    # was trained on a local PSG server; the dashboard may hit the NASA
    # remote API which can return a different absolute scale.  Band depths
    # (ratios) are unaffected, but the 154 raw-flux features need this.
    train_mean = _load_training_scale()
    if train_mean > 0:
        spec_mean = float(np.mean(interp_flux))
        if spec_mean > 0:
            interp_flux = interp_flux * (train_mean / spec_mean)

    X, _ = engineer_features(
        interp_flux.reshape(1, -1), training_wavelengths, include_raw=True
    )

    band_depths = extract_band_depths(interp_flux, training_wavelengths)
    diseq = extract_disequilibrium_features(band_depths)
    diagnostics = {**band_depths, **diseq}

    proba = model.predict_proba(X)[0]
    predicted_idx = int(np.argmax(proba))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(proba[predicted_idx])

    importances = model.feature_importances_
    top5_idx = np.argsort(importances)[-5:][::-1]
    key_features = [(feature_names[i], float(importances[i])) for i in top5_idx]

    if biosig_bands is not None:
        contributing_molecules = get_molecule_explanations(
            model, feature_names, training_wavelengths,
            biosig_bands, band_depths, predicted_class,
        )
    else:
        contributing_molecules = explain_prediction(
            interp_flux, training_wavelengths, predicted_class, band_depths
        )

    return {
        "class_names": CLASS_NAMES,
        "probabilities": [float(p) for p in proba],
        "predicted_class": predicted_class,
        "confidence": confidence,
        "is_inhabited": predicted_class in ("modern_earth", "archean_earth"),
        "key_features": key_features,
        "diagnostics": {k: float(v) for k, v in diagnostics.items()},
        "contributing_molecules": contributing_molecules,
        "class_description": CLASS_DESCRIPTIONS.get(predicted_idx, ""),
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def predict_from_spectrum_dict(spectrum_dict: dict) -> dict:
    """
    Predict from a spectrum dictionary (as returned by PSG service).

    Args:
        spectrum_dict: Dict with 'wavelength' and 'depth' (or 'flux') keys

    Returns:
        Prediction result dict
    """
    wavelengths = np.array(spectrum_dict["wavelength"])
    flux = np.array(spectrum_dict.get("depth", spectrum_dict.get("flux")))
    
    return predict(wavelengths, flux)


def get_class_info():
    """Return class names and descriptions for UI display."""
    return {
        "class_names": CLASS_NAMES,
        "descriptions": CLASS_DESCRIPTIONS,
    }


def get_model_info():
    """Return model metadata for UI display."""
    _, _, metadata = load_model()
    return {
        "n_features": metadata.get("n_features", 163),
        "n_classes": len(CLASS_NAMES),
        "wavelength_range": (metadata["wavelengths"][0], metadata["wavelengths"][-1]),
        "n_wavelengths": len(metadata["wavelengths"]),
    }