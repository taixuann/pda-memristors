
import os
import json
import subprocess
from pathlib import Path

# === Path utilities ===

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "raw_data"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_DIR = RESULTS_DIR / "summary"

def ensure_dirs():
    """Ensure results directories exist."""
    for d in [DATA_DIR, FIGURES_DIR, SUMMARY_DIR]:
        os.makedirs(d, exist_ok=True)

# === File save/load helpers ===

def save_json(data: dict, filename: str):
    """Save a dictionary as JSON inside results/data."""
    ensure_dirs()
    filepath = DATA_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    return filepath

def read_json(filepath: str):
    """Read JSON file and return dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)

# === Tagging utilities (macOS Finder tags) ===

def tag_file(filepath: str, *tags):
    """Apply macOS Finder tags to a file."""
    if not Path(filepath).exists():
        print(f"[Warning] File not found for tagging: {filepath}")
        return
    for t in tags:
        subprocess.run(["tag", "-a", t, str(filepath)], check=False)



