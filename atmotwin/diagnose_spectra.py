"""
Diagnostic round 2: test 'scl' (scale factor) units vs 'ppm' units with the 72-layer config.
"""
import time
from pathlib import Path

import numpy as np
import requests

PSG_URL = "https://psg.gsfc.nasa.gov/api.php"
REF_CONFIG = "modern_earth_LIFE_cfg.txt"

# Earth-default mixing ratios used as reference for scale factors
EARTH_DEFAULTS = {
    "H2O": 10000, "CO2": 400, "O3": 0.1, "N2O": 0.32,
    "CO": 0.1, "CH4": 1.8, "O2": 209000, "N2": 781000,
}


def call_psg(config_str, output_type="rad"):
    resp = requests.post(PSG_URL, data={"file": config_str, "type": output_type}, timeout=(10, 120))
    if resp.status_code != 200:
        raise Exception(f"PSG error: {resp.status_code}")
    wavelengths, flux = [], []
    for line in resp.text.strip().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                wavelengths.append(float(parts[0]))
                flux.append(float(parts[1]))
            except ValueError:
                continue
    if not wavelengths:
        raise Exception(f"No data. Snippet:\n{resp.text[:500]}")
    return np.array(wavelengths), np.array(flux)


def modify_atmosphere_scl(config_path, *, o2, ch4, co2, o3, n2o=0.0, co=0.0, h2o=0.0, n2=0.0):
    """Replace abundances using 'scl' (scale factor) units."""
    text = Path(config_path).read_text(encoding="utf-8", errors="replace")
    gases = ["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "N2"]
    desired = [h2o, co2, o3, n2o, co, ch4, o2, n2]
    defaults = [EARTH_DEFAULTS[g] for g in gases]
    scales = []
    for d, df in zip(desired, defaults):
        scales.append(d / df if df > 0 else 0.0)

    def _replace(t, tag, val):
        needle = f"<{tag}>"
        lines = t.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(needle):
                lines[i] = f"{needle}{val}"
                return "\n".join(lines) + ("\n" if t.endswith("\n") else "")
        return t + (("" if t.endswith("\n") else "\n") + f"{needle}{val}\n")

    text = _replace(text, "ATMOSPHERE-GAS", ",".join(gases))
    text = _replace(text, "ATMOSPHERE-ABUN", ",".join(f"{s:g}" for s in scales))
    text = _replace(text, "ATMOSPHERE-UNIT", ",".join(["scl"] * 8))
    return text


def modify_atmosphere_ppm(config_path, *, o2, ch4, co2, o3, n2o=0.0, co=0.0, h2o=0.0, n2=0.0):
    """Replace abundances using 'ppm' units (absolute mixing ratios)."""
    text = Path(config_path).read_text(encoding="utf-8", errors="replace")
    gases = ["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "N2"]
    abun = [h2o, co2, o3, n2o, co, ch4, o2, n2]

    def _replace(t, tag, val):
        needle = f"<{tag}>"
        lines = t.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(needle):
                lines[i] = f"{needle}{val}"
                return "\n".join(lines) + ("\n" if t.endswith("\n") else "")
        return t + (("" if t.endswith("\n") else "\n") + f"{needle}{val}\n")

    text = _replace(text, "ATMOSPHERE-GAS", ",".join(gases))
    text = _replace(text, "ATMOSPHERE-ABUN", ",".join(f"{v:g}" for v in abun))
    text = _replace(text, "ATMOSPHERE-UNIT", ",".join(["ppm"] * 8))
    return text


MODERN = dict(o2=210000, ch4=1.8, co2=400, h2o=10000, o3=0.1, n2o=0.32, co=0.1, n2=780000)
LIFELESS = dict(o2=0, ch4=0.001, co2=50000, h2o=10000, o3=0.0, n2o=0.0, co=0.1, n2=780000)

configs = [
    ("SCL-modern", modify_atmosphere_scl(REF_CONFIG, **MODERN)),
    ("SCL-lifeless", modify_atmosphere_scl(REF_CONFIG, **LIFELESS)),
    ("PPM-modern", modify_atmosphere_ppm(REF_CONFIG, **MODERN)),
    ("PPM-lifeless", modify_atmosphere_ppm(REF_CONFIG, **LIFELESS)),
]

KEY_WL = {"CO2 4.3µm": 4.3, "H2O 6.3µm": 6.3, "CH4 7.7µm": 7.7, "O3 9.6µm": 9.6, "CO2 15µm": 15.0}


def nearest(w, f, target):
    return f[np.argmin(np.abs(w - target))]


results = {}
for label, cfg in configs:
    print(f"Calling PSG: {label}...")
    w, f = call_psg(cfg)
    results[label] = (w, f)
    print(f"  {len(w)} points, flux [{f.min():.4e}, {f.max():.4e}]")
    time.sleep(5)

print("\n=== Flux at key wavelengths ===\n")
for feat, wl in KEY_WL.items():
    print(f"  {feat}:")
    for label, (w, f) in results.items():
        print(f"    {label:<20} {nearest(w, f, wl):.8e}")

print("\n=== Relative difference (modern vs lifeless) ===\n")
for feat, wl in KEY_WL.items():
    vs = nearest(*results["SCL-modern"], wl)
    vsl = nearest(*results["SCL-lifeless"], wl)
    vp = nearest(*results["PPM-modern"], wl)
    vpl = nearest(*results["PPM-lifeless"], wl)
    ds = abs(vs - vsl) / max(abs(vs), 1e-30)
    dp = abs(vp - vpl) / max(abs(vp), 1e-30)
    print(f"  {feat:<16}  scl: {ds:10.4%}    ppm: {dp:10.4%}")
