import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from plot_spectrum import call_psg_api


OUTPUT_CSV = "training_data.csv"
CHECKPOINT_JSON = "training_checkpoint.json"
ERROR_LOG = "generation_errors.log"

# Fixed wavelength grid for ML features (4.00, 4.05, ..., 18.50 µm)
WAVELENGTH_MIN = 4.0
WAVELENGTH_MAX = 18.5
WAVELENGTH_STEP = 0.05
WAVELENGTH_GRID = np.arange(WAVELENGTH_MIN, WAVELENGTH_MAX + 1e-6, WAVELENGTH_STEP)

TOTAL_PER_CLASS = 250


@dataclass
class ClassSpec:
    name: str
    pretty: str


CLASS_SPECS = {
    "inhabited_modern": ClassSpec(name="inhabited_modern", pretty="Modern Earth"),
    "inhabited_archean": ClassSpec(name="inhabited_archean", pretty="Archean Earth"),
    "lifeless": ClassSpec(name="lifeless", pretty="Lifeless"),
    "false_positive": ClassSpec(name="false_positive", pretty="False Positive (waterworld O₂)"),
}


def uniform_sample(low: float, high: float) -> float:
    return float(np.random.uniform(low, high))


def log_uniform_sample(low: float, high: float) -> float:
    """Sample log-uniformly between low and high (inclusive)."""
    low = float(low)
    high = float(high)
    if low <= 0.0 or high <= 0.0:
        raise ValueError("log_uniform_sample requires strictly positive bounds.")
    log_low = np.log(low)
    log_high = np.log(high)
    return float(np.exp(np.random.uniform(log_low, log_high)))


def build_psg_config(
    n2: float,
    o2: float,
    h2o: float,
    co2: float,
    ch4: float,
    co: float,
    o3: float,
    n2o: float,
) -> str:
    """
    Build a PSG configuration string using the provided atmospheric abundances.
    Follows the template given in the specification.
    """
    config = f"""<OBJECT>Exoplanet
<OBJECT-NAME>Earth-like
<OBJECT-DIAMETER>12742
<OBJECT-STAR-TYPE>G
<OBJECT-STAR-TEMPERATURE>5778
<OBJECT-STAR-RADIUS>1.0
<GEOMETRY>Observatory
<GEOMETRY-OBS-ALTITUDE>10
<GEOMETRY-ALTITUDE-UNIT>pc
<ATMOSPHERE-STRUCTURE>Equilibrium
<ATMOSPHERE-PRESSURE>1.0
<ATMOSPHERE-PUNIT>bar
<ATMOSPHERE-WEIGHT>28.97
<ATMOSPHERE>N2,O2,H2O,CO2,CH4,CO,O3,N2O
<ATMOSPHERE-NABNUNIT>ppmv
<ATMOSPHERE-NGAS>8
<ATMOSPHERE-GAS>N2,O2,H2O,CO2,CH4,CO,O3,N2O
<ATMOSPHERE-ABUN>{n2:.6g},{o2:.6g},{h2o:.6g},{co2:.6g},{ch4:.6g},{co:.6g},{o3:.6g},{n2o:.6g}
<ATMOSPHERE-UNIT>ppmv,ppmv,ppmv,ppmv,ppmv,ppmv,ppmv,ppmv
<GENERATOR-RANGE1>{WAVELENGTH_MIN}
<GENERATOR-RANGE2>{WAVELENGTH_MAX}
<GENERATOR-RESOLUTION>100
<GENERATOR-RESOLUTIONUNIT>RP
<GENERATOR-GAS-MODEL>Y
<GENERATOR-TRANS-APPLY>N
<GENERATOR-RADUNITS>Wm2um
"""
    return config


def call_psg_with_retries(config: str, max_retries: int = 3, delay_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call PSG via the shared call_psg_api helper with retries and delay.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            wavelength, flux = call_psg_api(config, output_type="rad")
            time.sleep(delay_s)
            return wavelength, flux
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            backoff = delay_s * attempt
            print(f"PSG call failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("PSG call failed without raising an exception.")


def validate_spectrum(wavelength: np.ndarray, flux: np.ndarray) -> bool:
    """
    Basic sanity checks on wavelength range and flux values.

    This is intentionally permissive to avoid rejecting otherwise valid PSG
    outputs. Detailed cleaning happens later when interpolating and using
    the spectra for ML.
    """
    if wavelength.size == 0 or flux.size == 0:
        return False
    # Require that the spectrum at least overlaps our target range.
    if wavelength.max() < WAVELENGTH_MIN or wavelength.min() > WAVELENGTH_MAX:
        return False
    if not np.any(np.isfinite(flux)):
        return False
    return True


def interpolate_to_grid(wavelength: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Interpolate spectrum onto the fixed wavelength grid."""
    return np.interp(WAVELENGTH_GRID, wavelength, flux, left=np.nan, right=np.nan)


def log_error(message: str) -> None:
    with Path(ERROR_LOG).open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_checkpoint() -> Dict[str, int]:
    if not Path(CHECKPOINT_JSON).exists():
        return {name: 0 for name in CLASS_SPECS}
    try:
        data = json.loads(Path(CHECKPOINT_JSON).read_text(encoding="utf-8"))
        for name in CLASS_SPECS:
            data.setdefault(name, 0)
        return {k: int(v) for k, v in data.items() if k in CLASS_SPECS}
    except Exception:
        return {name: 0 for name in CLASS_SPECS}


def save_checkpoint(counts: Dict[str, int]) -> None:
    Path(CHECKPOINT_JSON).write_text(json.dumps(counts, indent=2), encoding="utf-8")


def current_counts_from_csv(csv_path: Path) -> Dict[str, int]:
    if not csv_path.exists():
        return {name: 0 for name in CLASS_SPECS}
    try:
        df = pd.read_csv(csv_path, usecols=["label"])
    except Exception:
        return {name: 0 for name in CLASS_SPECS}
    counts = {name: 0 for name in CLASS_SPECS}
    for name in CLASS_SPECS:
        counts[name] = int((df["label"] == name).sum())
    return counts


def sample_parameters_for_class(class_name: str) -> Dict[str, float]:
    """
    Sample atmospheric parameters for a given class according to the spec.
    Returns a dict with keys: N2, O2, CH4, CO2, CO, H2O, N2O, O3.
    """
    if class_name == "inhabited_modern":
        n2 = 780000.0
        o2 = uniform_sample(150000.0, 210000.0)
        ch4 = log_uniform_sample(0.5, 5.0)
        co2 = uniform_sample(280.0, 560.0)
        co = log_uniform_sample(0.05, 0.3)
        h2o = uniform_sample(5000.0, 30000.0)
        n2o = uniform_sample(0.2, 0.5)
        o3 = log_uniform_sample(0.01, 1.0)
    elif class_name == "inhabited_archean":
        n2 = uniform_sample(780000.0, 900000.0)
        o2 = 0.0
        ch4 = log_uniform_sample(100.0, 10000.0)
        co2 = log_uniform_sample(165.0, 50000.0)
        co = log_uniform_sample(0.1, 50.0)
        h2o = uniform_sample(5000.0, 30000.0)
        n2o = 0.0
        o3 = 0.0
    elif class_name == "lifeless":
        n2 = uniform_sample(780000.0, 950000.0)
        o2 = 0.0
        ch4 = log_uniform_sample(0.001, 0.1)
        co2 = log_uniform_sample(100.0, 30000.0)
        co = log_uniform_sample(0.1, 10.0)
        h2o = log_uniform_sample(1000.0, 30000.0)
        n2o = 0.0
        o3 = 0.0
    elif class_name == "false_positive":
        n2 = 780000.0
        o2 = uniform_sample(100000.0, 210000.0)
        ch4 = log_uniform_sample(0.001, 0.1)
        co2 = log_uniform_sample(200.0, 5000.0)
        co = log_uniform_sample(50.0, 500.0)
        h2o = log_uniform_sample(100.0, 2000.0)
        n2o = 0.0
        o3 = log_uniform_sample(0.01, 0.3)
    else:
        raise ValueError(f"Unknown class name: {class_name}")

    return {
        "N2": n2,
        "O2": o2,
        "CH4": ch4,
        "CO2": co2,
        "CO": co,
        "H2O": h2o,
        "N2O": n2o,
        "O3": o3,
    }


def generate_training_data(
    output_csv: str = OUTPUT_CSV,
    max_retries: int = 3,
) -> Dict[str, int]:
    """
    Generate 1000 synthetic spectra (250 per class) using PSG.

    The CSV will have:
      - flux_x.xx columns for each wavelength bin on a fixed grid
      - molecule abundances (N2, O2, CH4, CO2, CO, H2O, N2O, O3)
      - label (class name)
    """
    out_path = Path(output_csv)

    # Determine starting counts from CSV and checkpoint (best-effort resume)
    csv_counts = current_counts_from_csv(out_path)
    ck_counts = load_checkpoint()
    counts = {name: max(csv_counts.get(name, 0), ck_counts.get(name, 0)) for name in CLASS_SPECS}

    total_target = TOTAL_PER_CLASS * len(CLASS_SPECS)
    already_done = sum(counts.values())
    if already_done > 0:
        print(f"Resuming from existing data: {already_done}/{total_target} samples already present.")

    # Prepare CSV header if new file
    if not out_path.exists():
        cols = [
            "label",
            "N2",
            "O2",
            "CH4",
            "CO2",
            "CO",
            "H2O",
            "N2O",
            "O3",
        ]
        for lam in WAVELENGTH_GRID:
            cols.append(f"flux_{lam:.2f}")
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)

    samples_since_save = 0

    while any(counts[name] < TOTAL_PER_CLASS for name in CLASS_SPECS):
        for class_name, spec in CLASS_SPECS.items():
            if counts[class_name] >= TOTAL_PER_CLASS:
                continue

            idx_in_class = counts[class_name] + 1
            overall_idx = sum(counts.values()) + 1
            print(f"Generating spectrum {overall_idx}/{total_target} ({spec.pretty}, {idx_in_class}/{TOTAL_PER_CLASS})...")

            params = sample_parameters_for_class(class_name)
            cfg = build_psg_config(
                n2=params["N2"],
                o2=params["O2"],
                h2o=params["H2O"],
                co2=params["CO2"],
                ch4=params["CH4"],
                co=params["CO"],
                o3=params["O3"],
                n2o=params["N2O"],
            )

            try:
                wavelength, flux = call_psg_with_retries(cfg, max_retries=max_retries, delay_s=1.0)
            except Exception as exc:  # noqa: BLE001
                msg = f"[{class_name}] PSG error for sample {idx_in_class}: {exc}"
                print(msg)
                log_error(msg)
                continue

            if not validate_spectrum(wavelength, flux):
                msg = f"[{class_name}] Invalid spectrum for sample {idx_in_class}; skipping."
                print(msg)
                log_error(msg)
                continue

            flux_grid = interpolate_to_grid(wavelength, flux)

            row = {
                "label": class_name,
                "N2": params["N2"],
                "O2": params["O2"],
                "CH4": params["CH4"],
                "CO2": params["CO2"],
                "CO": params["CO"],
                "H2O": params["H2O"],
                "N2O": params["N2O"],
                "O3": params["O3"],
            }
            for lam, fl in zip(WAVELENGTH_GRID, flux_grid):
                row[f"flux_{lam:.2f}"] = fl

            df_row = pd.DataFrame([row])
            df_row.to_csv(out_path, mode="a", header=False, index=False)

            counts[class_name] += 1
            samples_since_save += 1

            if samples_since_save >= 50:
                save_checkpoint(counts)
                samples_since_save = 0

            # Break out of the for-loop if we've reached the target overall
            if sum(counts.values()) >= total_target:
                break

    save_checkpoint(counts)
    return counts


def check_psg_running() -> bool:
    """Simple connectivity check using a minimal modern-Earth like atmosphere."""
    print("Checking PSG connection at local instance via call_psg_api...")
    try:
        cfg = build_psg_config(
            n2=780000.0,
            o2=210000.0,
            h2o=10000.0,
            co2=400.0,
            ch4=1.8,
            co=0.1,
            o3=0.3,
            n2o=0.33,
        )
        wavelength, flux = call_psg_with_retries(cfg, max_retries=1, delay_s=1.0)
        print(
            "PSG connectivity check response: "
            f"{len(wavelength)} points from {wavelength.min():.3f}–{wavelength.max():.3f} µm, "
            f"flux range [{flux.min():.3e}, {flux.max():.3e}]"
        )
        # Treat any successful response as "running"; detailed validation happens in the generator.
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"PSG connectivity check failed: {exc}")
        return False


if __name__ == "__main__":
    np.random.seed(int(time.time()) % (2**32 - 1))

    if not check_psg_running():
        raise SystemExit("PSG does not appear to be running or is misconfigured at http://localhost:3000/")

    start = time.time()
    final_counts = generate_training_data()
    duration_min = (time.time() - start) / 60.0

    total_generated = sum(final_counts.values())
    print("\n=== Generation summary ===")
    print(f"Total spectra generated: {total_generated}")
    for class_name, count in final_counts.items():
        spec = CLASS_SPECS[class_name]
        print(f"  {spec.pretty} ({class_name}): {count}")
    print(f"Total runtime: {duration_min:.1f} minutes (expected 30–60 minutes for full 1000 spectra).")

