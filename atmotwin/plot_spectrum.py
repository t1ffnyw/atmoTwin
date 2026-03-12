import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from typing import Optional, Tuple, Dict


USE_LOCAL_PSG = False
PSG_URL = "http://localhost:3000/api.php" if USE_LOCAL_PSG else "https://psg.gsfc.nasa.gov/api.php"

def call_psg_api(config_input, output_type='rad'):
    """
    Call PSG API and return spectrum data.
    
    Parameters:
    -----------
    config_input : str
        Either a path to a PSG configuration file or a configuration string
    output_type : str
        'rad' for radiance/emission, 'trn' for transmittance
    
    Returns:
    --------
    wavelength : numpy array
        Wavelength in microns
    spectrum : numpy array
        Flux or transmittance values
    """
    # Read config from file path or use as direct string
    if isinstance(config_input, str) and os.path.exists(config_input):
        with open(config_input, 'r') as f:
            config = f.read()
    else:
        config = str(config_input)
    
    # Send to PSG API
    response = requests.post(
        PSG_URL,
        data={'file': config, 'type': output_type}
    )
    
    if response.status_code != 200:
        raise Exception(f"PSG API error: {response.status_code}")
    
    # Parse the response
    lines = response.text.strip().split('\n')
    
    wavelengths = []
    flux = []
    
    for line in lines:
        # Skip comments and empty lines
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        
        # Parse data columns
        parts = line.split()
        if len(parts) >= 2:
            try:
                wavelengths.append(float(parts[0]))
                flux.append(float(parts[1]))  # 'Total' column
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


def create_base_config(
    o2: float = 210000,
    ch4: float = 1.8,
    co2: float = 400,
    co: float = 0.1,
    h2o: float = 10000,
    n2o: float = 0.32,
    o3: float = 0.1,
    wavelength_min: float = 4.0,
    wavelength_max: float = 18.5,
    resolution: float = 50,
) -> str:
    """
    Create a PSG configuration string with specified atmospheric parameters.
    Uses settings optimized for local PSG installation.
    """
    config = f"""<OBJECT>Planet
<OBJECT-NAME>Earth
<OBJECT-DATE>2020/04/08 01:32
<OBJECT-DIAMETER>12742
<OBJECT-GRAVITY>9.8
<OBJECT-GRAVITY-UNIT>g
<OBJECT-STAR-DISTANCE>1.0
<OBJECT-STAR-VELOCITY>0.0
<OBJECT-SOLAR-LONGITUDE>0.0
<OBJECT-SOLAR-LATITUDE>0.00
<OBJECT-SEASON>180.0
<OBJECT-STAR-TYPE>G
<OBJECT-STAR-TEMPERATURE>5777
<OBJECT-STAR-RADIUS>1.0
<OBJECT-OBS-LONGITUDE>180.00
<OBJECT-OBS-LATITUDE>0.00
<OBJECT-OBS-VELOCITY>0.0
<OBJECT-PERIOD>1.58040433
<OBJECT-STAR-METALLICITY>0.0
<OBJECT-INCLINATION>90
<GEOMETRY>Observatory
<GEOMETRY-OFFSET-NS>0.0
<GEOMETRY-OFFSET-EW>0.0
<GEOMETRY-OFFSET-UNIT>arcsec
<GEOMETRY-OBS-ALTITUDE>10
<GEOMETRY-ALTITUDE-UNIT>pc
<GEOMETRY-USER-PARAM>0.0
<GEOMETRY-STELLAR-TYPE>G
<GEOMETRY-STELLAR-TEMPERATURE>5777
<GEOMETRY-STELLAR-MAGNITUDE>0
<GEOMETRY-SOLAR-ANGLE>90.000
<GEOMETRY-OBS-ANGLE>48.121
<GEOMETRY-PLANET-FRACTION>1.000e+00
<GEOMETRY-STAR-DISTANCE>0.000000e+00
<GEOMETRY-STAR-FRACTION>8.386316e-05
<GEOMETRY-REF>User
<GEOMETRY-DISK-ANGLES>1
<ATMOSPHERE-DESCRIPTION>Custom atmosphere for AtmoTwin
<ATMOSPHERE-STRUCTURE>Equilibrium
<ATMOSPHERE-WEIGHT>28.97
<ATMOSPHERE-PRESSURE>1.00
<ATMOSPHERE-PUNIT>bar
<ATMOSPHERE-NGAS>8
<ATMOSPHERE-GAS>H2O,CO2,O3,N2O,CO,CH4,O2,N2
<ATMOSPHERE-TYPE>HIT[1],HIT[2],HIT[3],HIT[4],HIT[5],HIT[6],HIT[7],HIT[22]
<ATMOSPHERE-ABUN>{h2o},{co2},{o3},{n2o},{co},{ch4},{o2},780000
<ATMOSPHERE-UNIT>ppmv,ppmv,ppmv,ppmv,ppmv,ppmv,ppmv,ppmv
<GENERATOR-RANGE1>{wavelength_min}
<GENERATOR-RANGE2>{wavelength_max}
<GENERATOR-RANGEUNIT>um
<GENERATOR-RESOLUTION>{resolution}
<GENERATOR-RESOLUTIONUNIT>RP
<GENERATOR-GAS-MODEL>Y
<GENERATOR-CONT-MODEL>Y
<GENERATOR-CONT-STELLAR>N
<GENERATOR-TRANS-SHOW>N
<GENERATOR-RADUNITS>contrast
<GENERATOR-LOGRAD>N
<GENERATOR-TELESCOPE>SINGLE
<GENERATOR-DIAMTELE>1.0
<GENERATOR-NOISE>NO
<GENERATOR-BEAM>1.0
<GENERATOR-BEAM-UNIT>arcsec
"""
    return config

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
    Read a PSG config file and replace:
      - <ATMOSPHERE-GAS>
      - <ATMOSPHERE-ABUN>
      - <ATMOSPHERE-UNIT>
    using the provided mixing ratios (ppmv).
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
        # If not present, append at end (keeps file valid for PSG)
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


def generate_comparison_spectra(
    config_path: str,
    *,
    output_type: str = "rad",
    modern: Optional[Dict[str, float]] = None,
    lifeless: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Generate and plot "Modern Earth" vs "Lifeless Earth" spectra plus deviation.

    Modern defaults: O2=210000 ppmv, CH4=1.8 ppmv, CO2=400 ppmv, O3=0.1 ppmv
    Lifeless defaults: O2=0, CH4=0.001 ppmv, CO2=400 ppmv, O3=0
    """
    modern_params = {
        "o2_ppmv": 210000.0,
        "ch4_ppmv": 1.8,
        "co2_ppmv": 400.0,
        "o3_ppmv": 0.1,
        "n2o_ppmv": 0.0,
        "co_ppmv": 0.0,
        "h2o_ppmv": 0.0,
        "n2_ppmv": 0.0,
    }
    lifeless_params = {
        "o2_ppmv": 0.0,
        "ch4_ppmv": 0.001,
        "co2_ppmv": 400.0,
        "o3_ppmv": 0.0,
        "n2o_ppmv": 0.0,
        "co_ppmv": 0.0,
        "h2o_ppmv": 0.0,
        "n2_ppmv": 0.0,
    }
    if modern:
        modern_params.update(modern)
    if lifeless:
        lifeless_params.update(lifeless)

    modern_cfg = modify_atmosphere(config_path, **modern_params)
    lifeless_cfg = modify_atmosphere(config_path, **lifeless_params)

    # Use existing call_psg_api by writing temp config files.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f_mod:
        f_mod.write(modern_cfg)
        modern_cfg_path = f_mod.name
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f_life:
        f_life.write(lifeless_cfg)
        lifeless_cfg_path = f_life.name

    try:
        w_mod, y_mod = call_psg_api(modern_cfg_path, output_type=output_type)
        w_life, y_life = call_psg_api(lifeless_cfg_path, output_type=output_type)
    finally:
        # Best-effort cleanup (Windows needs files closed; they are).
        for p in (modern_cfg_path, lifeless_cfg_path):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    # Align grids if PSG returns slightly different sampling
    if len(w_mod) != len(w_life) or not np.allclose(w_mod, w_life, rtol=0, atol=0):
        y_life_interp = np.interp(w_mod, w_life, y_life, left=np.nan, right=np.nan)
        deviation = y_mod - y_life_interp
        w_common = w_mod
        y_life_common = y_life_interp
    else:
        w_common = w_mod
        y_life_common = y_life
        deviation = y_mod - y_life

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax0.plot(w_common, y_mod, label="Modern Earth", linewidth=1.0)
    ax0.plot(w_common, y_life_common, label="Lifeless Earth", linewidth=1.0)
    ax0.set_ylabel("Spectrum (PSG output)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(w_common, deviation, color="black", linewidth=1.0)
    ax1.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("Wavelength (µm)")
    ax1.set_ylabel("Modern − Lifeless")
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Modern vs Lifeless Earth (PSG)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig, (ax0, ax1)


def plot_spectrum(wavelength, flux, title="Earth Thermal Emission Spectrum"):
    """
    Plot the spectrum with labeled biosignature regions.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(wavelength, flux, 'b-', linewidth=0.8)
    ax.set_xlabel('Wavelength (µm)', fontsize=12)
    ax.set_ylabel('Contrast (Planet/Star)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark key biosignature wavelengths (single representative feature per label)
    biosignatures = {
        'CO₂': 4.3,
        'H₂O': 6.3,
        'CH₄': 7.7,
        'N₂O': 8.5,
        'O₃': 9.6,
        'CO₂ ': 15.0,  # Second CO₂ feature (note the space to differentiate)
    }

    colors = {
        'CO₂': 'red',
        'H₂O': 'blue',
        'CH₄': 'orange',
        'N₂O': 'green',
        'O₃': 'purple',
        'CO₂ ': 'red',
    }

    ymin, ymax = ax.get_ylim()
    # Add top headroom so biosignature labels are not clipped by the axes
    ax.set_ylim(ymin, ymax * 1.15)
    ymin, ymax = ax.get_ylim()

    #ax.set_ylim(ymin, 8.55e-5)
   # ymin, ymax = ax.get_ylim()

    label_heights = [0.86, 0.86, 0.86, 0.86, 0.86, 0.86]
    # Small, fixed horizontal offsets (in microns) to slightly separate nearby labels
    x_offsets = [-0.05, 0.05, -0.08, 0.08, -0.1, 0.1]

    for i, (molecule, w) in enumerate(biosignatures.items()):
        if not (wavelength.min() <= w <= wavelength.max()):
            continue

        ax.axvline(x=w, color=colors[molecule], linestyle='--', alpha=0.5, linewidth=1)

        height = label_heights[i % len(label_heights)]
        x_text = w + x_offsets[i % len(x_offsets)]
        ax.text(
            x_text,
            ymax * height,
            molecule,
            fontsize=9,
            color=colors[molecule],
            ha='center',
            va='bottom',
        )

    plt.tight_layout(pad=1.2)
    return fig, ax


def calculate_molecule_contributions(
    config_path: str,
    *,
    o2_ppmv: float = 210000.0,
    ch4_ppmv: float = 1.8,
    co2_ppmv: float = 400.0,
    o3_ppmv: float = 0.1,
    n2o_ppmv: float = 0.32,
    co_ppmv: float = 0.1,
    h2o_ppmv: float = 10000.0,
    n2_ppmv: float = 0.0,
    output_type: str = "rad",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute per-molecule contributions to the spectrum.

    A baseline spectrum is generated for the given atmosphere, then for each
    molecule a spectrum is generated with that molecule's abundance set to
    zero. The contribution is defined as:

        contribution = baseline_flux - flux_without_molecule

    Returns the common wavelength grid and a dictionary that maps each
    molecule name to its contribution array.
    """
    # Build baseline configuration and spectrum
    baseline_cfg = modify_atmosphere(
        config_path,
        o2_ppmv=o2_ppmv,
        ch4_ppmv=ch4_ppmv,
        co2_ppmv=co2_ppmv,
        o3_ppmv=o3_ppmv,
        n2o_ppmv=n2o_ppmv,
        co_ppmv=co_ppmv,
        h2o_ppmv=h2o_ppmv,
        n2_ppmv=n2_ppmv,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f_base:
        f_base.write(baseline_cfg)
        baseline_cfg_path = f_base.name

    try:
        w_base, y_base = call_psg_api(baseline_cfg_path, output_type=output_type)
    finally:
        try:
            Path(baseline_cfg_path).unlink(missing_ok=True)
        except Exception:
            pass

    # Define molecules and their corresponding abundance parameters
    molecules = [
        ("O3", "o3_ppmv"),
        ("CH4", "ch4_ppmv"),
        ("CO2", "co2_ppmv"),
        ("H2O", "h2o_ppmv"),
        ("N2O", "n2o_ppmv"),
        ("CO", "co_ppmv"),
        ("O2", "o2_ppmv"),
    ]

    base_params = {
        "o2_ppmv": o2_ppmv,
        "ch4_ppmv": ch4_ppmv,
        "co2_ppmv": co2_ppmv,
        "o3_ppmv": o3_ppmv,
        "n2o_ppmv": n2o_ppmv,
        "co_ppmv": co_ppmv,
        "h2o_ppmv": h2o_ppmv,
        "n2_ppmv": n2_ppmv,
    }

    contributions: Dict[str, np.ndarray] = {}

    for mol_name, param_key in molecules:
        params_without = dict(base_params)
        params_without[param_key] = 0.0

        cfg_without = modify_atmosphere(config_path, **params_without)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f_m:
            f_m.write(cfg_without)
            cfg_without_path = f_m.name

        try:
            w_m, y_m = call_psg_api(cfg_without_path, output_type=output_type)
        finally:
            try:
                Path(cfg_without_path).unlink(missing_ok=True)
            except Exception:
                pass

        # Align to the baseline wavelength grid if needed
        if len(w_m) != len(w_base) or not np.allclose(w_m, w_base, rtol=0, atol=0):
            y_m_interp = np.interp(w_base, w_m, y_m, left=np.nan, right=np.nan)
        else:
            y_m_interp = y_m

        contrib = y_base - y_m_interp
        contributions[mol_name] = contrib

    return w_base, contributions


def plot_molecule_contributions(
    wavelength: np.ndarray,
    contributions: Dict[str, np.ndarray],
    *,
    stacked: bool = True,
    title: str = "Per-molecule contributions to the spectrum",
):
    """
    Plot per-molecule contributions as a stacked area or multi-line chart.

    Parameters
    ----------
    wavelength : array
        Common wavelength grid.
    contributions : dict
        Maps molecule name (e.g., 'O3', 'CH4') to contribution array.
    stacked : bool
        If True, uses a stacked area plot. If False, uses separate lines
        with semi-transparent fills.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    plot_order = [
        ("O3", "O₃"),
        ("CH4", "CH₄"),
        ("CO2", "CO₂"),
        ("H2O", "H₂O"),
        ("N2O", "N₂O"),
        ("CO", "CO"),
        ("O2", "O₂"),
    ]

    colors = {
        "O3": "purple",
        "CH4": "orange",
        "CO2": "red",
        "H2O": "blue",
        "N2O": "green",
        "CO": "brown",
        "O2": "magenta",
    }

    arrays = []
    labels = []
    color_list = []

    for key, pretty in plot_order:
        if key not in contributions:
            continue
        arr = np.asarray(contributions[key])
        # Clip small negative values from numerical noise so stackplot behaves well
        arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0)
        arrays.append(arr)
        labels.append(pretty)
        color_list.append(colors.get(key, None))

    if not arrays:
        return fig, ax

    if stacked:
        ax.stackplot(wavelength, arrays, labels=labels, colors=color_list, alpha=0.8)
    else:
        for arr, lbl, col in zip(arrays, labels, color_list):
            ax.plot(wavelength, arr, label=lbl, color=col, linewidth=1.0)
            ax.fill_between(wavelength, 0, arr, color=col, alpha=0.3)

    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Contribution (baseline - no molecule)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout(pad=1.0)

    return fig, ax


# Main execution
if __name__ == "__main__":
    # Test with your config file
    config_file = 'modern_earth_LIFE_cfg.txt'
    
    print("Calling PSG API...")
    wavelength, flux = call_psg_api(config_file, output_type='rad')
    
    print(f"Received {len(wavelength)} spectral points")
    print(f"Wavelength range: {wavelength.min():.2f} - {wavelength.max():.2f} µm")
    
    # Plot
    fig, ax = plot_spectrum(wavelength, flux, title="Modern Earth - Thermal Emission (LIFE Range)")
    plt.savefig('modern_earth_spectrum.png', dpi=150)
    plt.show()
    
    print("Spectrum saved to 'modern_earth_spectrum.png'")

    # Comparison plot (Modern vs Lifeless)
    fig2, _ = generate_comparison_spectra(
        config_file,
        output_type="rad",
        save_path="modern_vs_lifeless_comparison.png",
    )
    plt.show()