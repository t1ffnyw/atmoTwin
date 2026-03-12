from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


WAVELENGTH_MIN = 4.0
WAVELENGTH_MAX = 18.33
WAVELENGTH_STEP = 0.5


def build_wavelength_grid() -> np.ndarray:
    """
    Build the wavelength grid:
    4.0, 4.5, ..., 18.0, 18.33 (µm).
    """
    base = np.arange(WAVELENGTH_MIN, 18.0 + 1e-6, WAVELENGTH_STEP)
    # Ensure the final point is exactly 18.33 µm
    if np.isclose(base[-1], WAVELENGTH_MAX, atol=1e-6):
        return base
    return np.concatenate([base, np.array([WAVELENGTH_MAX])])


def synthetic_flux_curve(wavelengths: np.ndarray) -> np.ndarray:
    """
    Generate a simple, smooth synthetic flux curve whose values
    are in the same ballpark as those in training_data.csv
    (~1e-1 down to a few 1e-4).
    """
    wl = np.asarray(wavelengths, dtype=float)
    t = np.linspace(0.0, 1.0, wl.size)
    flux_max = 0.12
    flux_min = 4.4e-4
    # Log-linear decay from flux_max to flux_min
    return flux_max * (flux_min / flux_max) ** t


def wavelength_flux_rows() -> Iterable[Tuple[float, float]]:
    """Yield (wavelength, flux) pairs for CSV writing."""
    wavelengths = build_wavelength_grid()
    fluxes = synthetic_flux_curve(wavelengths)
    for wl, fl in zip(wavelengths, fluxes):
        yield float(wl), float(fl)


def generate_wl_flux_csv(output_path: str | Path | None = None) -> Path:
    """
    Create a small CSV file with:

        wavelength,flux
        4.00,0.11647
        4.50,0.07553
        ...
        18.33,0.00044

    The exact values are synthetic but live in the same numeric
    range as the flux values used in the ML training data.
    """
    if output_path is None:
        output_path = Path(__file__).parent / "test_wavelength_flux.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("wavelength,flux\n")
        for wl, fl in wavelength_flux_rows():
            f.write(f"{wl:.2f},{fl:.5f}\n")

    return output_path


if __name__ == "__main__":
    csv_path = generate_wl_flux_csv()
    print(f"Test wavelength/flux CSV written to: {csv_path}")

