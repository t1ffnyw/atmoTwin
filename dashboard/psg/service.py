"""
Generate a spectrum from dashboard planet params (gases in ppmv) via PSG.
"""
from . import CONFIG_PATH
from .client import call_psg_api, modify_atmosphere


def generate_spectrum(planet_params: dict, output_type: str = "rad") -> dict:
    """
    Build PSG config from dashboard planet params and return spectrum.

    planet_params: dict from get_planet_params() with keys star_type, orbital_distance_au,
                   surface_temp_k, surface_pressure_bar, gases (dict of ppmv values).
    output_type: 'rad' for radiance (default), 'trn' for transmittance.

    Returns:
        {"wavelength": np.array (μm), "depth": np.array} — "depth" is flux/radiance or transmittance.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"PSG base config not found: {CONFIG_PATH}")

    gases = planet_params.get("gases", {})

    def ppmv(gas: str, default: float = 0.0) -> float:
        return float(gases.get(gas, default))

    cfg_str = modify_atmosphere(
        str(CONFIG_PATH),
        o2_ppmv=ppmv("O2", 210000.0),
        ch4_ppmv=ppmv("CH4", 1.8),
        co2_ppmv=ppmv("CO2", 400.0),
        o3_ppmv=ppmv("O3", 0.1),
        n2o_ppmv=ppmv("N2O", 0.32),
        co_ppmv=ppmv("CO", 0.1),
        h2o_ppmv=ppmv("H2O", 10000.0),
        n2_ppmv=ppmv("N2", 780000.0),
    )

    wavelength, flux = call_psg_api(cfg_str, output_type=output_type)
    return {"wavelength": wavelength, "depth": flux}
