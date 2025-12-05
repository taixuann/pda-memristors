import numpy as np, pandas as pd
import os, json, subprocess
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path


# === Path utilities ===

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "raw_data"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_DIR = RESULTS_DIR / "summary"
LOG_DIR = RESULTS_DIR / "log"
SVG_DIR = RESULTS_DIR / "svg"

# === Read CV file ===

def read_data_cv(cv_path):
    """
    Read the cv data with current and potential file
    """
    # --- Input path ---
    expected_cols = ['Potential applied (V)', 'Time (s)', 'WE(1).Current (A)', 'WE(1).Potential (V)', 'Scan', 'Index', 'Q+', 'Q-']
    data = pd.read_csv(cv_path, sep = ";", decimal = ",", header = 0, usecols = expected_cols)
    # --- Read potential and current ---
    potential = data['Potential applied (V)'].values
    current = data['WE(1).Current (A)'].values
    # --- Read cycles ----
    cycles = []
    scans = data['Scan'].values
    uniq_scans = pd.unique(scans)
    for s in uniq_scans:
        idxs = np.where(scans == s)[0]
        if idxs.size >0:
            cycles.append((int(idxs[0]), int(idxs[-1] + 1)))
                          
    num_cycles = len(cycles)
    return potential, current, cycles, num_cycles

# === Plotting utilities ===

def plot_style(ax: plt.Axes, legend_title_case: bool = True):
    """
    Apply a standard style to a matplotlib plot.

    This function styles the plot according to the following guidelines:
    - Uses a sans-serif font for axis labels and titles.
    - Sets font size for labels and ticks between 8 and 14 points.
    - Positions the legend within the figure boundaries.
    - Capitalizes legend text to title case.

    Args:
        ax (plt.Axes): The matplotlib axes object to style.
        legend_title_case (bool): If True, converts legend labels to title case.
    """
    font_config = {'family': 'sans-serif', 'size': 12}
    
    ax.set_xlabel(ax.get_xlabel(), **font_config)
    ax.set_ylabel(ax.get_ylabel(), **font_config)
    ax.tick_params(axis='both', which='major', labelsize=font_config['size'])
    title_font_config = font_config.copy()
    del title_font_config['size']
    ax.set_title(ax.get_title(), **title_font_config, size=14)

    if ax.get_legend():
        ax.legend(loc='best')
        if legend_title_case:
            for text in ax.get_legend().get_texts():
                text.set_text(text.get_text().title())

def color_selection_plot(color_name: str) -> str:
    """
    Provides a centralized color palette for plots. Call a color by its name.

    Args:
        color_name (str): The desired color's name (e.g., 'red', 'blue', 'anodic_peak').

    Returns:
        str: The hex code for the requested color. Defaults to black if not found.
    """
    palette = {
        # --- Standard Colors ---
        'red': '#d62728',       # Matplotlib default red
        'blue': '#1f77b4',      # Matplotlib default blue
        'green': '#2ca02c',     # Matplotlib default green
        'orange': '#ff7f0e',    # Matplotlib default orange
        'purple': '#9467bd',    # Matplotlib default purple
        'black': '#000000',
        'gray': '#7f7f7f',
        'light_gray': '#c7c7c7',

        # --- Semantic Colors for CV Plots ---
        'anodic_peak': '#d62728',
        'cathodic_peak': '#1f77b4',
    }
    
    color_code = palette.get(color_name.lower(), '#000000') # Default to black
    return color_code
    