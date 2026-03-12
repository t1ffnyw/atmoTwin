HERO_CONTENT = {
    "title": "AtmoTwin: Exoplanet Atmosphere Digital Twin",
    "tagline": "Build, simulate, and interpret Earth-like exoplanet atmospheres for biosignature detection.",
    "problem_statement": (
    "We cannot directly observe life on distant worlds — but we can study the chemistry of their atmospheres. Atmospheric characterization of terrestrial exoplanets is one of the central goals of astrobiology: searching for signs of biological activity, assessing habitability, and exploring the diversity of planetary atmospheres across the galaxy."
    "The LIFE (Large Interferometer For Exoplanets) mission concept proposes a space-based observatory operating in the mid-infrared to detect and investigate the thermal emission spectra of rocky exoplanets. This approach offers unique scientific potential for identifying biosignatures, evaluating habitability, and advancing comparative planetology."
    "\n"
    "\nAtmoTwin puts that science in your hands. Build an exoplanet atmosphere, simulate what LIFE would see, and let a machine-learning classifier tell you whether the signal points to a living world, a lifeless one, or a false positive."
    ),
    "baseline_scenario": "Modern Earth analog orbiting a G-type star at 1 AU.",
    "wavelength_range": "Near-IR to mid-IR transmission spectrum (4.0–18.3 μm).",
    "primary_application": "Biosignature detection and atmospheric characterization from low-S/N spectra.",
}


WORKFLOW_STEPS = [
    {
        "id": "builder",
        "label": "Builder",
        "step_number": 1,
        "headline": "Configure the planet–star system",
        "summary": "Choose a preset (e.g., Modern Earth) or dial in custom gases, star type, and orbit.",
        "detail": (
            "Use the Builder page to define your host star, orbital distance, and atmospheric composition. "
            "Start from curated presets or construct your own scenario to test specific hypotheses about "
            "biosignatures and false positives."
        ),
        "page_path": "pages/1_Builder.py",
        "cta_label": "Open Builder",
    },
    {
        "id": "results",
        "label": "Results",
        "step_number": 2,
        "headline": "Generate spectra and classifications",
        "summary": "Run the forward model and inspect the synthetic transmission spectrum and ML classification.",
        "detail": (
            "After running a simulation from Builder, the Spectrum and Classifier pages show the resulting "
            "transmission spectrum, key biosignature bands, and an ML-based assessment of atmospheric "
            "disequilibrium, including potential false-positive flags."
        ),
        "page_path": "pages/2_Results.py",
        "cta_label": "View results",
    },
    {
        "id": "explore",
        "label": "Explore",
        "step_number": 3,
        "headline": "Compare and iterate on scenarios",
        "summary": "Save interesting simulations, revisit them, and compare side by side.",
        "detail": (
            "Use the Explorer page to store multiple scenarios, quickly toggle between them, and compare how "
            "changes in composition or stellar environment move you between biological, abiotic, and "
            "false-positive regimes."
        ),
        "page_path": "pages/3_Explorer.py",
        "cta_label": "Explore scenarios",
    },
]


RESOURCES = [
    {
        "label": "GitHub repository",
        "emoji": "💻",
        "description": "Source code, issues, and contribution guide.",
        # Replace this placeholder with your actual repository URL.
        "url": "https://github.com/t1ffnyw/atmoTwin",
    },
    {
        "label": "Documentation",
        "emoji": "📘",
        "description": "User guide, API docs, and examples.",
        "url": "https://docs.google.com/document/d/1Y8ull0PBBkkRc0IUgysQXNgJQ9VuIaxE3ecp1eRsQrA/edit?usp=sharing",
    },
    {
        "label": "References",
        "emoji": "📄",
        "description": "Methods and validation for the AtmoTwin pipeline.",
        "url": "https://life-space-mission.com/",
    },
]

