hack-4-sages / **AtmoTwin Dashboard**

AtmoTwin is an interactive Streamlit dashboard for building, simulating, and interpreting exoplanet atmospheres using LIFE-like thermal emission spectra. Users can compose atmospheres, call a PSG-based forward model, and run a machine-learning classifier to explore inhabited, lifeless, and false-positive scenarios.

---

## Quickstart

- **Create environment**
  - `python -m venv .venv`
  - Activate it (PowerShell): `.\.venv\Scripts\Activate.ps1`
  - Install deps: `pip install -r requirements.txt`
- **Run the dashboard**
  - From the repo root: `streamlit run dashboard/Home.py`
  - Use the Streamlit sidebar to navigate between pages.

You need PSG configured and reachable for the spectrum-generation endpoints used by the dashboard (see `atmotwin/` notes below).

---

## Project layout

```text
dashboard/
├── Home.py                 # Streamlit home / landing page
├── config.py               # Star presets, gases, biosignature bands, scenario presets
├── state.py                # Shared Streamlit session state helpers
├── ui.py                   # Global page config + shared LIFE CSS theme
├── home_content.py         # Text, workflow steps, and external links for Home
├── components/
│   ├── planet_controls.py  # Host star + orbital + gas composition widgets
│   ├── spectrum_plot.py    # Plotly spectrum, contributions, comparison figures
│   └── result_cards.py     # Classification cards, probabilities, false-positive flags
└── pages/
    ├── 1_Builder.py        # Atmosphere builder + CSV upload + PSG + classifier
    ├── 2_Results.py        # Spectrum visualisation, per-molecule contributions, RF output
    └── 3_Explorer.py       # Scenario saving and side‑by‑side spectrum comparison

atmotwin/
├── atmotwin_app.py         # Original single-page AtmoTwin prototype (legacy)
├── data_loader.py          # Training/evaluation data utilities
├── inference.py            # Model loading + prediction helpers
├── train_model.py          # Training script for the AtmoTwin classifier
├── plot_spectrum*.py       # PSG API helpers and plotting utilities
└── *.json / *.csv / *.txt  # Model metadata, training data, PSG config templates

model/
├── data_loader.py          # Dashboard-facing data loader
├── inference.py            # Dashboard-facing classifier wrapper (used in Builder/Results)
└── __init__.py
```

---

## Dashboard pages

- **Home (`dashboard/Home.py`)**
  - Uses `configure_page()` from `ui.py` and structured content from `home_content.py`.
  - Explains the problem AtmoTwin addresses, provides a 3-step workflow (Builder → Results → Explorer), and links out to external resources.

- **Atmosphere Builder (`dashboard/pages/1_Builder.py`)**
  - Scenario presets come from `SCENARIO_PRESETS` in `config.py` (Modern Earth, Lifeless, Archean, Volcanic/Prebiotic, Ocean Loss).
  - **Atmosphere Composer** tab:
    - Lets you adjust atmospheric gases via `render_gas_inputs()` from `components/planet_controls.py`.
    - Calls `psg.service.generate_spectrum()` and `psg.service.calculate_contributions()` using `get_planet_params()` from `state.py`.
    - Runs `model.inference.predict()` on the resulting spectrum and stores `spectrum`, `classification`, `false_positive_flags`, and `contributions` in `st.session_state`.
  - **Upload Spectrum** tab:
    - Validates a CSV with columns `wavelength, flux, error`, stores it in `st.session_state.uploaded_builder_spectrum`, and shows a quick-look spectrum using `make_spectrum_figure()`.
    - Supports downstream analysis on the Results page.

- **Results (`dashboard/pages/2_Results.py`)**
  - **Spectrum** tab:
    - Lets you choose between the simulated spectrum and an uploaded spectrum.
    - Plots the spectrum with optional biosignature markers and log/linear y-axis via `make_spectrum_figure()`.
    - When using a simulated spectrum, plots per-molecule contributions with `make_contributions_figure()` using `st.session_state.contributions`.
  - **Classification** tab:
    - Shows RF classifier output for the simulated spectrum and/or uploaded spectrum using:
      - `render_upload_classification()` and `render_classification_card()` from `components/result_cards.py`.
      - Inline explanations of how features and disequilibrium ratios inform the label (inhabited / lifeless / false positive).

- **Scenario Explorer (`dashboard/pages/3_Explorer.py`)**
  - Allows saving the current scenario (`get_planet_params()`) into `st.session_state.saved_scenarios`.
  - Lets you rename or clear scenarios and then compare any two by calling `psg.service.generate_comparison_spectra()`.
  - Visualises side‑by‑side spectra and a deviation panel via `make_comparison_figure()` in `components/spectrum_plot.py`.

---

## Core modules

- **`config.py`**
  - Defines:
    - `STAR_PRESETS`: host star effective temperatures and luminosities.
    - `GASES`: labels, default ppmv, slider ranges/steps for key molecules (O₂, CH₄, CO₂, CO, H₂O, N₂O, O₃, N₂).
    - `BIOSIG_BANDS`: canonical biosignature band centers and colors for plotting.
    - `SCENARIO_PRESETS`: pre-defined atmospheres for Modern Earth, Lifeless Earth, Archean Earth, Volcanic/Prebiotic, and Ocean Loss.

- **`state.py`**
  - `init_state()`: sets up all required `st.session_state` keys (planet parameters, spectra, classification, flags, uploaded spectrum, presets, saved scenarios).
  - `get_planet_params()`: serialises the current scenario into a dict used by PSG service calls.
  - `load_preset(name)`: overwrites state with a named scenario and syncs gas input widgets.

- **`components/planet_controls.py`**
  - `render_star_selector()` and `render_orbital_params()` render high-level system parameters.
  - `render_gas_inputs()` renders text-based gas composition inputs and keeps them in sync with `st.session_state.gases`.

- **`components/spectrum_plot.py`**
  - `make_spectrum_figure()`: plots the main thermal emission spectrum with LIFE-like styling and molecular markers.
  - `make_contributions_figure()`: stacked area chart for per-molecule contributions.
  - `make_comparison_figure()`: dual-panel comparison of two scenarios (overlay + deviation).

- **`components/result_cards.py`**
  - `render_upload_classification()`: 4-class RF classification display for uploaded/simulated spectra (Modern, Archean, Lifeless, False Positive), with probability bars and diagnostic features.
  - `render_classification_card()`: simpler equilibrium/disequilibrium summary card.
  - `render_false_positive_warnings()`: lists potential astrophysical/photochemical false-positive flags.

---

## Atmotwin backend and model

- The `atmotwin/` package contains the original single-page AtmoTwin Streamlit app (`atmotwin_app.py`) plus PSG interfacing utilities and training scripts.
- The `model/` package provides the classifier and data-loading functions used by the dashboard.
- PSG configuration templates (e.g., `modern_earth_LIFE_cfg.txt`) and metadata files (`model_metadata.json`, `training_data.csv`, etc.) live alongside these modules and must be present and correctly referenced for forward-model calls and classification to work.

---

## Development tips

- The dashboard assumes it is run from the project root so that intra-package imports (`psg`, `model`, `atmotwin`) resolve correctly.
- If you modify presets or gases, keep `config.py`, `state.py`, and `components/planet_controls.py` in sync.
- For classifier changes, update the training scripts in `atmotwin/`, regenerate the model artifact, and adjust `model/inference.py` as needed.