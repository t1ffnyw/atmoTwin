import requests
import numpy as np
from scipy.stats import qmc
import time
import json
from pathlib import Path

PSG_URL = "http://localhost:3000/api.php"
OUTPUT_DIR = Path("psg_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# LITERATURE-VERIFIED ATMOSPHERIC RANGES BY CLASS
# Sources: Meadows 2023, Ulses 2025, Krissansen-Totton 2021
# =============================================================================

CLASS_DEFINITIONS = {
    'modern_earth': {
        'label': 0,
        'description': 'Inhabited (Modern) - Oxygenic photosynthesis biosphere',
        'ranges': {
            'O2':  (150000, 210000),    # 15-21%
            'CH4': (0.5, 5),             # Modern ~1.9 ppm
            'CO2': (280, 560),           # Pre-industrial to 2x
            'CO':  (0.05, 0.3),          # Low in oxic atmosphere
            'H2O': (5000, 30000),        # Variable humidity
            'N2O': (0.2, 0.5),           # ~0.33 modern
            'O3':  (0.01, 1),            # Stratospheric
        }
    },
    'archean_earth': {
        'label': 1,
        'description': 'Inhabited (Archean) - Methanogenic biosphere, no O2',
        'ranges': {
            'O2':  (0, 0),               # Pre-Great Oxidation Event (FIXED at 0)
            'CH4': (100, 10000),         # Methanogenic biosphere
            'CO2': (165, 50000),         # Up to 5%
            'CO':  (0.1, 50),            # Higher than modern
            'H2O': (5000, 30000),        # Similar to modern
            'N2O': (0, 0),               # Limited without O2 cycle (FIXED at 0)
            'O3':  (0, 0),               # No O2 source (FIXED at 0)
        }
    },
    'lifeless': {
        'label': 2,
        'description': 'Lifeless - Chemical equilibrium atmosphere',
        'ranges': {
            'O2':  (0, 0),               # No photosynthesis (FIXED at 0)
            'CH4': (0.001, 0.1),         # Abiotic trace only
            'CO2': (100, 30000),         # Volcanic outgassing
            'CO':  (0.1, 10),            # Equilibrium
            'H2O': (1000, 30000),        # Variable
            'N2O': (0, 0),               # Biogenic only (FIXED at 0)
            'O3':  (0, 0),               # No O2 (FIXED at 0)
        }
    },
    'false_positive': {
        'label': 3,
        'description': 'False Positive - Abiotic O2 (waterworld scenario)',
        'ranges': {
            'O2':  (100000, 210000),     # Abiotic from H2O photolysis
            'CH4': (0.001, 0.1),         # No biology
            'CO2': (200, 5000),          # Variable
            'CO':  (50, 500),            # Elevated (KEY INDICATOR)
            'H2O': (100, 2000),          # Surface depleted (KEY INDICATOR)
            'N2O': (0, 0),               # No biology (FIXED at 0)
            'O3':  (0.01, 0.3),          # Some from abiotic O2
        }
    }
}

# Fixed background gas
N2_ABUNDANCE = 780000  # ppm

# Gas order for PSG config
GAS_ORDER = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'N2']
HITRAN_IDS = 'HIT[1],HIT[2],HIT[3],HIT[4],HIT[5],HIT[6],HIT[7],HIT[22]'

# =============================================================================
# LATIN HYPERCUBE SAMPLING PER CLASS
# =============================================================================

def generate_class_samples(class_name, n_samples, seed=None):
    """Generate LHS samples for a specific atmospheric class"""
    
    class_def = CLASS_DEFINITIONS[class_name]
    ranges = class_def['ranges']
    
    # Identify which parameters are fixed vs variable
    variable_params = []
    fixed_params = {}
    
    for gas, (min_val, max_val) in ranges.items():
        if min_val == max_val:
            fixed_params[gas] = min_val
        else:
            variable_params.append(gas)
    
    n_variable = len(variable_params)
    
    if n_variable == 0:
        # All parameters fixed - just return copies
        samples = []
        for _ in range(n_samples):
            sample = fixed_params.copy()
            sample['N2'] = N2_ABUNDANCE
            samples.append(sample)
        return samples
    
    # Latin Hypercube for variable parameters
    sampler = qmc.LatinHypercube(d=n_variable, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    
    samples = []
    for i in range(n_samples):
        sample = fixed_params.copy()
        
        for j, gas in enumerate(variable_params):
            min_val, max_val = ranges[gas]
            
            # Use log sampling for gases with large ranges
            if max_val / max(min_val, 1e-10) > 100:
                log_min = np.log10(max(min_val, 1e-10))
                log_max = np.log10(max_val)
                sample[gas] = 10 ** (log_min + unit_samples[i, j] * (log_max - log_min))
            else:
                sample[gas] = min_val + unit_samples[i, j] * (max_val - min_val)
        
        sample['N2'] = N2_ABUNDANCE
        samples.append(sample)
    
    return samples

# =============================================================================
# PSG INTERFACE
# =============================================================================

def make_config(params):
    """Generate PSG config from parameter dict"""
    abundances = ','.join([f"{params[g]:.6f}" for g in GAS_ORDER])
    units = ','.join(['ppm'] * len(GAS_ORDER))
    
    return f"""<OBJECT>Exoplanet
<OBJECT-STAR-TYPE>G
<OBJECT-STAR-TEMP>5780
<ATMOSPHERE-DESCRIPTION>Custom
<ATMOSPHERE-STRUCTURE>Equilibrium
<ATMOSPHERE-PRESSURE>1
<ATMOSPHERE-PUNIT>bar
<ATMOSPHERE-WEIGHT>28.97
<ATMOSPHERE-NGAS>{len(GAS_ORDER)}
<ATMOSPHERE-GAS>{','.join(GAS_ORDER)}
<ATMOSPHERE-TYPE>{HITRAN_IDS}
<ATMOSPHERE-ABUN>{abundances}
<ATMOSPHERE-UNIT>{units}
<GENERATOR-RANGE1>4
<GENERATOR-RANGE2>18.5
<GENERATOR-RESOLUTION>100
<GENERATOR-RESOLUTIONUNIT>RP
<GENERATOR-GAS-MODEL>Y"""

def parse_spectrum(text):
    """Parse PSG output to wavelength and flux arrays"""
    wavelengths = []
    flux = []
    for line in text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                wavelengths.append(float(parts[0]))
                flux.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(wavelengths), np.array(flux)

def run_psg(config, retries=3):
    """Send config to PSG with retry logic"""
    for attempt in range(retries):
        try:
            r = requests.post(PSG_URL, data={'file': config}, timeout=30)
            if r.status_code != 200:
                continue
            if '# ERROR' in r.text:
                return None, None, r.text
            w, f = parse_spectrum(r.text)
            if len(w) > 0:
                return w, f, None
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1)
            continue
    return None, None, "Max retries exceeded"

# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================

def generate_dataset(samples_per_class=100):
    """Generate balanced dataset with all 4 classes"""
    
    total_samples = samples_per_class * len(CLASS_DEFINITIONS)
    
    print("=" * 70)
    print("AtmoTwin Training Data Generation")
    print("=" * 70)
    print(f"Samples per class: {samples_per_class}")
    print(f"Total samples: {total_samples}")
    print(f"Classes: {list(CLASS_DEFINITIONS.keys())}")
    print("=" * 70)
    
    all_spectra = []
    all_params = []
    all_labels = []
    all_class_names = []
    wavelengths = None
    errors = []
    
    start_time = time.time()
    sample_count = 0
    
    for class_name, class_def in CLASS_DEFINITIONS.items():
        print(f"\n{'='*70}")
        print(f"Generating: {class_def['description']}")
        print(f"{'='*70}")
        
        # Show ranges for this class
        print("Parameter ranges:")
        for gas, (min_v, max_v) in class_def['ranges'].items():
            if min_v == max_v:
                print(f"  {gas}: {min_v} (fixed)")
            else:
                print(f"  {gas}: {min_v} - {max_v} ppm")
        
        # Generate samples for this class
        seed = hash(class_name) % (2**32)
        samples = generate_class_samples(class_name, samples_per_class, seed=seed)
        
        print(f"\n{'#':<4} {'H2O':>8} {'CO2':>8} {'O3':>6} {'CH4':>8} {'O2':>10} {'CO':>6} {'Time':>6}")
        print("-" * 60)
        
        for i, sample in enumerate(samples):
            config = make_config(sample)
            
            t0 = time.time()
            w, f, err = run_psg(config)
            elapsed = time.time() - t0
            
            sample_count += 1
            
            if err:
                errors.append((class_name, i, err[:100]))
                print(f"{i+1:<4} ❌ Error: {err[:50]}...")
                continue
            
            if wavelengths is None:
                wavelengths = w
            
            # Store results
            all_spectra.append(f)
            all_params.append([sample[g] for g in GAS_ORDER[:-1]])  # Exclude N2
            all_labels.append(class_def['label'])
            all_class_names.append(class_name)
            
            # Progress output
            if (i + 1) % 10 == 0 or i == 0:
                print(f"{i+1:<4} {sample['H2O']:>8.0f} {sample['CO2']:>8.1f} "
                      f"{sample['O3']:>6.3f} {sample['CH4']:>8.3f} "
                      f"{sample['O2']:>10.0f} {sample['CO']:>6.2f} {elapsed:>5.2f}s")
        
        # Class summary
        class_count = sum(1 for c in all_class_names if c == class_name)
        print(f"\n✅ {class_name}: {class_count}/{samples_per_class} successful")
    
    # =============================================================================
    # SAVE DATASET
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("Saving dataset...")
    print("=" * 70)
    
    spectra_array = np.array(all_spectra)
    params_array = np.array(all_params)
    labels_array = np.array(all_labels)
    
    # 1. NumPy format (for ML training)
    np.savez(
        OUTPUT_DIR / "atmotwin_training_data.npz",
        wavelengths=wavelengths,
        spectra=spectra_array,
        params=params_array,
        labels=labels_array,
        class_names=list(CLASS_DEFINITIONS.keys()),
        gas_names=GAS_ORDER[:-1]  # Exclude N2
    )
    print(f"✅ Saved: atmotwin_training_data.npz")
    print(f"   - Spectra shape: {spectra_array.shape}")
    print(f"   - Labels shape: {labels_array.shape}")
    
    # 2. CSV for inspection
    import csv
    
    # Combined CSV with labels
    with open(OUTPUT_DIR / "training_data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header: class, label, params, spectrum
        header = ['class_name', 'label'] + GAS_ORDER[:-1] + [f"wl_{w:.4f}" for w in wavelengths]
        writer.writerow(header)
        
        for i in range(len(all_spectra)):
            row = [all_class_names[i], all_labels[i]] + list(params_array[i]) + list(spectra_array[i])
            writer.writerow(row)
    
    print(f"✅ Saved: training_data.csv")
    
    # 3. Metadata
    metadata = {
        'n_samples': len(all_spectra),
        'n_wavelengths': len(wavelengths),
        'samples_per_class': {
            name: sum(1 for c in all_class_names if c == name) 
            for name in CLASS_DEFINITIONS.keys()
        },
        'wavelength_range_um': [float(wavelengths[0]), float(wavelengths[-1])],
        'spectral_resolution': 100,
        'flux_unit': 'W/sr/m2/um',
        'classes': {
            name: {
                'label': cdef['label'],
                'description': cdef['description'],
                'ranges': cdef['ranges']
            }
            for name, cdef in CLASS_DEFINITIONS.items()
        },
        'gas_order': GAS_ORDER[:-1],
        'sources': [
            'Meadows et al. 2023',
            'Ulses et al. 2025', 
            'Krissansen-Totton et al. 2021'
        ]
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved: metadata.json")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total samples: {len(all_spectra)}")
    print(f"Total errors: {len(errors)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nClass distribution:")
    for name in CLASS_DEFINITIONS.keys():
        count = sum(1 for c in all_class_names if c == name)
        print(f"  {name}: {count}")
    
    if errors:
        print(f"\n⚠️ Errors:")
        for cls, idx, err in errors[:5]:
            print(f"  {cls}[{idx}]: {err[:60]}...")
    
    print("\n✅ Ready for Random Forest training!")
    print(f"   Load with: data = np.load('psg_dataset/atmotwin_training_data.npz')")

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    generate_dataset(samples_per_class=250)  # 1000 total samples