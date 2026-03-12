# PSG (Planetary Spectrum Generator) integration for dashboard
from pathlib import Path

from .client import call_psg_api, modify_atmosphere

# Base config path: next to this package
CONFIG_PATH = Path(__file__).resolve().parent / "modern_earth_LIFE_cfg.txt"

__all__ = ["call_psg_api", "modify_atmosphere", "CONFIG_PATH"]
