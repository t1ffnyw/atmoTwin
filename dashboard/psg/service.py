"""
Generate a spectrum from dashboard planet params (gases in ppmv) via PSG.
"""
from typing import Dict, List, Tuple

import numpy as np

from . import CONFIG_PATH
from .client import call_psg_api, modify_atmosphere

MOLECULE_PARAM_MAP = [
    ("O3", "o3_ppmv"),
    ("CH4", "ch4_ppmv"),
    ("CO2", "co2_ppmv"),
    ("H2O", "h2o_ppmv"),
    ("N2O", "n2o_ppmv"),
    ("CO", "co_ppmv"),
    ("O2", "o2_ppmv"),
]


def _gas_params(planet_params: dict) -> dict:
    gases = planet_params.get("gases", {})

    def ppmv(gas: str, default: float = 0.0) -> float:
        return float(gases.get(gas, default))

    return dict(
        o2_ppmv=ppmv("O2", 210000.0),
        ch4_ppmv=ppmv("CH4", 1.8),
        co2_ppmv=ppmv("CO2", 400.0),
        o3_ppmv=ppmv("O3", 0.1),
        n2o_ppmv=ppmv("N2O", 0.32),
        co_ppmv=ppmv("CO", 0.1),
        h2o_ppmv=ppmv("H2O", 10000.0),
        n2_ppmv=ppmv("N2", 780000.0),
    )


def generate_spectrum(planet_params: dict, output_type: str = "rad") -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"PSG base config not found: {CONFIG_PATH}")

    cfg_str = modify_atmosphere(str(CONFIG_PATH), **_gas_params(planet_params))
    wavelength, flux = call_psg_api(cfg_str, output_type=output_type)
    return {"wavelength": wavelength, "depth": flux}


def calculate_contributions(
    planet_params: dict, output_type: str = "rad"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    For each molecule, generate a spectrum with that molecule zeroed out,
    then subtract from the baseline to get its contribution.

    Returns (wavelength, baseline_flux, {molecule_name: contribution_array}).
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"PSG base config not found: {CONFIG_PATH}")

    base_kw = _gas_params(planet_params)
    cfg_path = str(CONFIG_PATH)

    baseline_cfg = modify_atmosphere(cfg_path, **base_kw)
    w_base, y_base = call_psg_api(baseline_cfg, output_type=output_type)

    contributions: Dict[str, np.ndarray] = {}
    for mol_name, param_key in MOLECULE_PARAM_MAP:
        if base_kw.get(param_key, 0.0) == 0.0:
            contributions[mol_name] = np.zeros_like(y_base)
            continue

        without_kw = dict(base_kw)
        without_kw[param_key] = 0.0
        cfg_without = modify_atmosphere(cfg_path, **without_kw)
        w_m, y_m = call_psg_api(cfg_without, output_type=output_type)

        if len(w_m) != len(w_base) or not np.allclose(w_m, w_base, atol=0):
            y_m = np.interp(w_base, w_m, y_m, left=np.nan, right=np.nan)

        contributions[mol_name] = np.maximum(y_base - y_m, 0.0)

    return w_base, y_base, contributions


def generate_comparison_spectra(
    scenarios: List[dict],
    output_type: str = "rad",
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate spectra for an arbitrary list of planet-param dicts and align
    them to a common wavelength grid.

    Returns (wavelength, [flux_array_per_scenario]).
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"PSG base config not found: {CONFIG_PATH}")

    cfg_path = str(CONFIG_PATH)
    results: List[Tuple[np.ndarray, np.ndarray]] = []

    for params in scenarios:
        cfg_str = modify_atmosphere(cfg_path, **_gas_params(params))
        w, y = call_psg_api(cfg_str, output_type=output_type)
        results.append((w, y))

    w_common = results[0][0]
    aligned_fluxes: List[np.ndarray] = [results[0][1]]

    for w_i, y_i in results[1:]:
        if len(w_i) != len(w_common) or not np.allclose(w_i, w_common, atol=0):
            y_i = np.interp(w_common, w_i, y_i, left=np.nan, right=np.nan)
        aligned_fluxes.append(y_i)

    return w_common, aligned_fluxes
