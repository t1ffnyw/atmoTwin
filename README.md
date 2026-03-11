# hack-4-sages

## File structure

```
dashboard/
├── app.py                    # Entry point → Home page
├── config.py                 # Constants and presets
├── state.py                  # Session state and preset loading
├── ui.py                     # Shared page config helper
├── pages/
│   ├── 1_Builder.py          # Atmosphere builder + simulate
│   ├── 2_Spectrum.py        # Spectrum viewer
│   ├── 3_Classifier.py      # Classification results
│   └── 4_Explorer.py        # Scenario save/compare
└── components/
    ├── planet_controls.py   # Star/orbit/gas inputs
    ├── spectrum_plot.py     # Plotly spectrum + band overlays
    └── result_cards.py      # Classification and false-positive UI
```

---

## Purpose of each file

| File | Purpose |
| --- | --- |
| [**app.py**](http://app.py/) | **Home.** Entry point and landing page: title, short app description, “What you can do here” (Builder, Spectrum, Classifier, Explorer), suggested workflow, and a note to use the sidebar to switch pages. No simulation logic. |
| [**config.py**](http://config.py/) | Shared data: `STAR_PRESETS`, `GASES` (labels, defaults, ranges), `BIOSIG_BANDS` (wavelength/color per band), and `SCENARIO_PRESETS` (e.g. Modern Earth, Archean, Volcanic, Ocean Loss). |
| [**state.py**](http://state.py/) | Session state: `init_state()` sets defaults (planet params, spectrum, classification, saved_scenarios); `get_planet_params()` reads current config; `load_preset(name)` applies a scenario and clears results. |
| [**ui.py**](http://ui.py/) | Single helper `configure_page()` that calls `st.set_page_config` (title “AtmoTwin”, icon, wide layout, expanded sidebar). Used by `app.py` and every page so the browser tab and layout are consistent. |
| **pages/1_Builder.py** | **Builder.** Two columns: left = spectrum + classification tabs (and Simulate logic); right = planet parameters (star, orbit, pressure, gas sliders) and “Simulate” button. Sidebar: “Quick Start” preset selector. Runs the placeholder spectrum/classifier and writes to `st.session_state`. |
| **pages/2_Spectrum.py** | **Spectrum.** Shows the current `st.session_state.spectrum` with `make_spectrum_figure` and a “Highlight biosignature bands” toggle. If no spectrum, shows a short message to run a simulation (e.g. from Builder). |
| **pages/3_Classifier.py** | **Classifier.** Shows `st.session_state.classification` via `render_classification_card` and `render_false_positive_warnings`. If no classification, tells the user to run a simulation. |
| **pages/4_Explorer.py** | **Explorer.** Save current scenario (from `get_planet_params()`) into `saved_scenarios`, clear all saved scenarios, and list saved scenarios in expanders (JSON). Preset dropdown to load a preset into current state. |
| **components/planet_controls.py** | Reusable UI: `render_star_selector()`, `render_orbital_params()` (distance, temp, pressure), `render_gas_sliders()` (log₁₀ mixing ratios). All bind to `st.session_state` keys used by `state.py`. |
| **components/spectrum_plot.py** | `make_spectrum_figure(spectrum, show_bands)` builds a Plotly transmission spectrum and optionally draws biosignature band regions from `BIOSIG_BANDS` using current gas mixing ratios. |
| **components/result_cards.py** | `render_classification_card(result)` shows label (disequilibrium/equilibrium), confidence bar, and key features; `render_false_positive_warnings(flags)` shows warning/error messages for false-positive flags. |

---

**Flow:** Home explains the app → Builder is where you set planet params and run Simulate → Spectrum and Classifier show the last run’s results from session state → Explorer stores and compares scenario snapshots and can load presets.