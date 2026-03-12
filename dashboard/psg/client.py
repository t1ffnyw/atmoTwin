"""
PSG API client: call NASA (or local) Planetary Spectrum Generator and modify atmosphere config.
"""
import os
from pathlib import Path

import numpy as np
import requests

# Use NASA remote API by default so the app works without a local PSG server
USE_LOCAL_PSG = False
PSG_URL = "http://localhost:3000/api.php" if USE_LOCAL_PSG else "https://psg.gsfc.nasa.gov/api.php"


def call_psg_api(config_input, output_type="rad"):
    """
    Call PSG API and return spectrum data.

    config_input: path to a config file (str) or config string.
    output_type: 'rad' for radiance/emission, 'trn' for transmittance.

    Returns:
        wavelength: np.array (μm)
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
