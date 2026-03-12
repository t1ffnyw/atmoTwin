import json
from pathlib import Path

import joblib
import numpy as np

from .data_loader import (
    engineer_features,
    extract_band_depths,
    extract_disequilibrium_features,
)

_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODEL_DIR.parent
_ATMOTWIN_DIR = _PROJECT_ROOT / "atmotwin"

# ── constants ─────────────────────────────────────────────────────────────────

_WL_MIN = 4.0  # µm — lower bound of training grid
_WL_MAX = 18.33  # µm — upper bound of training grid (last feature bin)
_WL_TOLERANCE = 0.5  # µm — how much the user's range can fall short


# ── model loading ──────────────────────────────────────────────────────────────

_MODEL_CACHE: tuple | None = None  # (model, feature_names, metadata)


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

    with (_ATMOTWIN_DIR / "feature_names.json").open() as f:
        feature_names = json.load(f)

    with (_ATMOTWIN_DIR / "model_metadata.json").open() as f:
        metadata = json.load(f)

    _MODEL_CACHE = (model, feature_names, metadata)
    return _MODEL_CACHE


# ── interpolation ──────────────────────────────────────────────────────────────


def interpolate_to_training_grid(
    user_wavelengths: np.ndarray,
    user_flux: np.ndarray,
    training_wavelengths: np.ndarray,
) -> np.ndarray:
    """Interpolate uploaded spectrum to the model's exact wavelength grid.

    Args:
        user_wavelengths: 1D array, sorted ascending (µm)
        user_flux: 1D array of flux values matching user_wavelengths
        training_wavelengths: 1D array of target wavelength bins

    Returns:
        np.ndarray of interpolated flux values, shape (n_wavelengths,)

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
    """Run the full inference pipeline on a spectrum.

    Args:
        user_wavelengths: 1D array of wavelength values (µm), sorted ascending
        user_flux: 1D array of flux/contrast values

    Returns:
        dict with keys:
            class_names    list[str]
            probabilities  list[float]
            predicted_class str
            confidence     float
            is_inhabited   bool
            key_features   list[(str,float)]
            diagnostics    dict[str,float]
    """
    model, feature_names, metadata = load_model()

    # Use the exact wavelength grid the model was trained on so that engineered
    # features at inference match training-time features.
    training_wavelengths = np.array(metadata["wavelengths"])

    interp_flux = interpolate_to_training_grid(
        np.asarray(user_wavelengths, dtype=float),
        np.asarray(user_flux, dtype=float),
        training_wavelengths,
    )

    # Build full feature vector: raw bins + engineered features
    X, _ = engineer_features(
        interp_flux.reshape(1, -1), training_wavelengths, include_raw=True
    )  # shape (1, 163)

    # Instance-level engineered diagnostics for UI display
    band_depths = extract_band_depths(interp_flux, training_wavelengths)
    diseq = extract_disequilibrium_features(band_depths)
    diagnostics = {**band_depths, **diseq}

    class_names = metadata["class_names"]
    proba = model.predict_proba(X)[0]  # shape (4,)
    predicted_idx = int(np.argmax(proba))
    predicted_class = class_names[predicted_idx]
    confidence = float(proba[predicted_idx])

    importances = model.feature_importances_  # shape (n_features,)
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

