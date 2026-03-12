"""
PSG API client: call NASA (or local) Planetary Spectrum Generator.

Two config-building strategies:
  - modify_atmosphere(): patches an existing base config file (richer
    atmospheric model, used for display / graphing).
  - make_config(): builds a minimal config from scratch matching the
    format used by generate_training_data.py (used for the RF classifier).
"""
import os
from pathlib import Path

import numpy as np
import requests

USE_LOCAL_PSG = False
PSG_URL = "http://localhost:3000/api.php" if USE_LOCAL_PSG else "https://psg.gsfc.nasa.gov/api.php"

GAS_ORDER = ["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "N2"]
HITRAN_IDS = "HIT[1],HIT[2],HIT[3],HIT[4],HIT[5],HIT[6],HIT[7],HIT[22]"


# ── Config for the RF classifier (from-scratch) ────────────────────────────────

def make_config(
    *,
    h2o_ppmv: float = 10000.0,
    co2_ppmv: float = 400.0,
    o3_ppmv: float = 0.1,
    n2o_ppmv: float = 0.32,
    co_ppmv: float = 0.1,
    ch4_ppmv: float = 1.8,
    o2_ppmv: float = 210000.0,
    n2_ppmv: float = 780000.0,
) -> str:
    """
    Build a clean PSG config from scratch -- identical format to the one used
    by generate_training_data.py so the RF classifier sees spectra in the same
    representation it was trained on.

    No pre-computed atmosphere layers: PSG computes the atmosphere in
    equilibrium from the provided ppmv values.
    """
    abun = [h2o_ppmv, co2_ppmv, o3_ppmv, n2o_ppmv, co_ppmv, ch4_ppmv, o2_ppmv, n2_ppmv]
    abundances = ",".join(f"{v:.6f}" for v in abun)
    units = ",".join(["ppm"] * len(GAS_ORDER))

    return (
        f"<OBJECT>Exoplanet\n"
        f"<OBJECT-STAR-TYPE>G\n"
        f"<OBJECT-STAR-TEMP>5780\n"
        f"<ATMOSPHERE-DESCRIPTION>Custom\n"
        f"<ATMOSPHERE-STRUCTURE>Equilibrium\n"
        f"<ATMOSPHERE-PRESSURE>1\n"
        f"<ATMOSPHERE-PUNIT>bar\n"
        f"<ATMOSPHERE-WEIGHT>28.97\n"
        f"<ATMOSPHERE-NGAS>{len(GAS_ORDER)}\n"
        f"<ATMOSPHERE-GAS>{','.join(GAS_ORDER)}\n"
        f"<ATMOSPHERE-TYPE>{HITRAN_IDS}\n"
        f"<ATMOSPHERE-ABUN>{abundances}\n"
        f"<ATMOSPHERE-UNIT>{units}\n"
        f"<GENERATOR-RANGE1>4\n"
        f"<GENERATOR-RANGE2>18.5\n"
        f"<GENERATOR-RESOLUTION>100\n"
        f"<GENERATOR-RESOLUTIONUNIT>RP\n"
        f"<GENERATOR-GAS-MODEL>Y"
    )


# ── Config for graphing (patch base config file) ───────────────────────────────

def modify_atmosphere(
    config_path: str,
    *,
    o2_ppmv: float,
    ch4_ppmv: float,
    co2_ppmv: float,
    o3_ppmv: float,
    n2o_ppmv: float = 0.0,
    co_ppmv: float = 0.0,
    h2o_ppmv: float = 0.0,
    n2_ppmv: float = 0.0,
) -> str:
    """
    Read a PSG config file and replace ATMOSPHERE-GAS, ATMOSPHERE-ABUN, ATMOSPHERE-UNIT
    with the given mixing ratios (ppmv). Returns the modified config string.
    """
    config_text = Path(config_path).read_text(encoding="utf-8", errors="replace")

    gases = ["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "N2"]
    abun_ppmv = [h2o_ppmv, co2_ppmv, o3_ppmv, n2o_ppmv, co_ppmv, ch4_ppmv, o2_ppmv, n2_ppmv]
    units = ["ppm"] * len(gases)

    def _replace_line(text: str, tag: str, value: str) -> str:
        needle = f"<{tag}>"
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(needle):
                lines[i] = f"{needle}{value}"
                return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        suffix = "" if text.endswith("\n") else "\n"
        return text + suffix + f"{needle}{value}\n"

    config_text = _replace_line(config_text, "ATMOSPHERE-GAS", ",".join(gases))
    config_text = _replace_line(
        config_text,
        "ATMOSPHERE-ABUN",
        ",".join(f"{v:g}" for v in abun_ppmv),
    )
    config_text = _replace_line(config_text, "ATMOSPHERE-UNIT", ",".join(units))

    return config_text


# ── PSG API call ────────────────────────────────────────────────────────────────

def call_psg_api(config_input, output_type="rad"):
    """
    Call PSG API and return spectrum data.

    config_input: path to a config file (str) or config string.
    output_type: 'rad' for radiance/emission, 'trn' for transmittance.

    Returns:
        wavelength: np.array (um)
        spectrum: np.array (flux or transmittance)
    """
    if isinstance(config_input, str) and os.path.exists(config_input):
        with open(config_input, "r", encoding="utf-8") as f:
            config = f.read()
    else:
        config = str(config_input)

    response = requests.post(PSG_URL, data={"file": config, "type": output_type}, timeout=60)

    if response.status_code != 200:
        raise Exception(f"PSG API error: {response.status_code}")

    lines = response.text.strip().split("\n")
    wavelengths = []
    flux = []

    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                wavelengths.append(float(parts[0]))
                flux.append(float(parts[1]))
            except ValueError:
                continue

    if len(wavelengths) == 0:
        snippet = response.text[:800].replace("\r", "")
        raise Exception(
            "PSG API returned no numeric spectrum rows. "
            "This is often a transient PSG error/throttle. "
            f"Response snippet:\n{snippet}"
        )

    return np.array(wavelengths), np.array(flux)
