import json
import sys
from pathlib import Path

import joblib
import numpy as np

_ATMOTWIN_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ATMOTWIN_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from atmotwin.data_loader import engineer_features

# ── constants ─────────────────────────────────────────────────────────────────

_WL_MIN = 4.0    # µm — lower bound of training grid
_WL_MAX = 18.33  # µm — upper bound of training grid (last feature bin)
_WL_TOLERANCE = 0.5  # µm — how much the user's range can fall short

# ── model loading ──────────────────────────────────────────────────────────────

_MODEL_CACHE = None  # (model, feature_names, metadata)

def load_model():
    """Load trained RF model, feature names, and metadata from disk.

    Returns:
        model: fitted RandomForestClassifier
        feature_names: list of 163 feature name strings
        metadata: dict from model_metadata.json
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    model = joblib.load(_ATMOTWIN_DIR / "atmotwin_classifier.joblib")

    with open(_ATMOTWIN_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(_ATMOTWIN_DIR / "model_metadata.json") as f:
        metadata = json.load(f)

    _MODEL_CACHE = (model, feature_names, metadata)
    return _MODEL_CACHE


# ── interpolation ──────────────────────────────────────────────────────────────

def interpolate_to_training_grid(user_wavelengths, user_flux, training_wavelengths):
    """Interpolate uploaded spectrum to the model's exact wavelength grid.

    Args:
        user_wavelengths: 1D array, sorted ascending (µm)
        user_flux: 1D array of flux values matching user_wavelengths
        training_wavelengths: 1D array of target wavelength bins

    Returns:
        np.ndarray of interpolated flux values, shape (154,)

    Raises:
        ValueError: if user wavelengths don't cover the required range
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


# ── inference ──────────────────────────────────────────────────────────────────

def predict(user_wavelengths, user_flux):
    """Run the full inference pipeline on an uploaded spectrum.

    Args:
        user_wavelengths: 1D array of wavelength values (µm), sorted ascending
        user_flux: 1D array of flux/contrast values

    Returns:
        dict with keys:
            class_names    list[str]        — 4 class labels in model order
            probabilities  list[float]      — predicted probability for each class
            predicted_class str             — class with highest probability
            confidence     float            — max probability (0–1)
            is_inhabited   bool             — True for modern_earth or archean_earth
            key_features   list[(str,float)]— top 5 (feature_name, importance) pairs
            diagnostics    dict[str,float]  — engineered diagnostic features for this spectrum
    """
    model, feature_names, metadata = load_model()

    # Use the exact wavelength grid the model was trained on to ensure that
    # the engineered features at inference match training-time features.
    training_wavelengths = np.array(metadata["wavelengths"])

    interp_flux = interpolate_to_training_grid(
        user_wavelengths, user_flux, training_wavelengths
    )

    # Build 163-feature vector: 154 raw bins + 9 engineered features
    X, _ = engineer_features(
        interp_flux.reshape(1, -1), training_wavelengths, include_raw=True
    )  # shape (1, 163)

    # Recompute engineered diagnostic features as a dict for UI display.
    # This mirrors what engineer_features() appends to the end of X.
    from atmotwin.data_loader import extract_band_depths, extract_disequilibrium_features

    band_depths = extract_band_depths(interp_flux, training_wavelengths)
    diseq = extract_disequilibrium_features(band_depths)
    diagnostics = {**band_depths, **diseq}

    class_names = metadata["class_names"]
    proba = model.predict_proba(X)[0]  # shape (4,)
    predicted_idx = int(np.argmax(proba))
    predicted_class = class_names[predicted_idx]
    confidence = float(proba[predicted_idx])

    importances = model.feature_importances_  # shape (163,)
    top5_idx = np.argsort(importances)[-5:][::-1]
    key_features = [(feature_names[i], float(importances[i])) for i in top5_idx]

    return {
        "class_names": class_names,
        "probabilities": [float(p) for p in proba],
        "predicted_class": predicted_class,
        "confidence": confidence,
        "is_inhabited": predicted_class in ("modern_earth", "archean_earth"),
        "key_features": key_features,
        "diagnostics": {k: float(v) for k, v in diagnostics.items()},
    }
