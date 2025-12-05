from pathlib import Path
import matplotlib.pyplot as plt
from modules import utils
from scipy.signal import find_peaks
import numpy as np, pandas as pd, os
from . import raw_cv_support

def construct_multiple_paths(scan_rates, filename_template, data_dir=utils.RAW_DATA, **kwargs):
    """
    Constructs a list of file paths for multiple CV files based on scan rates.

    Args:
        scan_rates (list): A list of scan rates (e.g., [10, 20, 50]).
        filename_template (str): A string that defines the file naming pattern,
                                 using placeholders like {scan_rate}, {ph}, etc.
                                 Example: "PDA_{electrodes}_{scan_rate}mV_pH {ph}.txt"
        data_dir (Path or str, optional): The directory containing the data files.
                                          Defaults to the global RAW_DATA directory.
        **kwargs: Additional keyword arguments to format the filename_template
                  (e.g., electrodes='Au, Pt, SCE', ph=8.4).

    Returns:
        list: A list of full Path objects for each CV file.
    """
    paths = []
    for rate in scan_rates:
        # Create a copy of kwargs to avoid modifying it in place
        format_args = kwargs.copy()
        format_args['scan_rate'] = rate
        paths.append(Path(data_dir) / filename_template.format(**format_args))
        filename = filename_template.format(**format_args)
        # Replace spaces with underscores for filesystem compatibility
    return paths

def construct_single_path(scan_rate, filename_template, data_dir=utils.RAW_DATA, **kwargs):
    """
    Constructs a single file path for a CV file based on a scan rate.

    Args:
        scan_rate (int or float): The scan rate (e.g., 100).
        filename_template (str): A string that defines the file naming pattern.
        data_dir (Path or str, optional): The directory containing the data file.
        **kwargs: Additional keyword arguments to format the filename_template.

    Returns:
        Path: A full Path object for the CV file.
    """
    format_args = kwargs.copy()
    format_args['scan_rate'] = scan_rate
    path = Path(data_dir) / filename_template.format(**format_args)
    filename = filename_template.format(**format_args)
    return path

def first_cv_with_peaks(cv_path, cycle_number=0):
    """
    Plots a single CV cycle and highlights the detected anodic and cathodic peaks.

    Args:
        cv_path (str or Path): The path to the CV data file.
        cycle_number (int): The cycle number to plot (default is 0 for the first cycle).
    """
    potential, current, cycles, _ = utils.read_data_cv(cv_path)
    peak_data = raw_cv_support.save_peaks_cv(cv_path)

    if cycle_number >= len(cycles):
        print(f"[Warning] Cycle {cycle_number + 1} not found in {Path(cv_path).name}. Skipping peak plot.")
        return

    start, end = cycles[cycle_number]
    cycle_peaks = peak_data.iloc[cycle_number]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(potential[start:end], current[start:end], label=f'Cycle {cycle_number + 1}')
    ax.scatter(cycle_peaks['Anodic_E(V)'], cycle_peaks['Anodic_I(A)'], color='red', s=80, zorder=5, label='Anodic Peak')
    # Plot both cathodic peaks if they exist
    ax.scatter(cycle_peaks['Cathodic_1_E(V)'], cycle_peaks['Cathodic_1_I(A)'], color='green', s=80, zorder=5, label='Cathodic Peak 1')
    ax.scatter(cycle_peaks['Cathodic_2_E(V)'], cycle_peaks['Cathodic_2_I(A)'], color='purple', s=80, zorder=5, label='Cathodic Peak 2')

    ax.set_title(f"CV with Detected Peaks - {Path(cv_path).stem}")
    ax.set_xlabel("Potential (V vs SCE)")
    ax.set_ylabel("Current (A)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    utils.plot_style(ax)
    
    fig_file = utils.LOG_DIR / f"{Path(cv_path).stem}_cycle{cycle_number+1}_peaks.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"✔ Peak verification plot saved to {fig_file.name}")

# === Plot all CVs with peaks for verification ===

def plot_all_cvs_with_peaks(cv_paths, scan_rates_mv, file_save, cycle_number=0):
    """
    Plots multiple CV cycles from different files on a single figure,
    highlighting the detected peaks for each.

    Args:
        cv_paths (list): A list of paths to the CV data files.
        scan_rates_mv (list): A list of scan rates corresponding to the files.
        file_save (str): The base name for the saved figure file.
        cycle_number (int): The cycle number to plot (default is 0).
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(cv_paths)))

    for i, cv_path in enumerate(cv_paths):
        scan_rate = scan_rates_mv[i]
        color = colors[i]

        potential, current, cycles, _ = utils.read_data_cv(cv_path)
        peak_data = raw_cv_support.save_peaks_cv(cv_path)

        if cycle_number >= len(cycles):
            print(f"[Warning] Cycle {cycle_number + 1} not found in {Path(cv_path).name}. Skipping.")
            continue

        start, end = cycles[cycle_number]
        # Plot CV curve
        plt.plot(potential[start:end], current[start:end], label=f'{scan_rate} mV/s', color=color)

    plt.title(f"Cycle {cycle_number + 1}")
    plt.xlabel("Potential (V vs SCE)")
    plt.ylabel("Current (A)")
    plt.legend(title="Scan Rate")
    plt.grid(True, linestyle='--', alpha=0.6)
    fig_file = utils.SUMMARY_DIR / f"{file_save}_all_peaks_verification.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"✔ Combined peak verification plot saved to {fig_file.name}")

def regression_line(sqrt_scan_rates, anodic_peaks_uA, cathodic_peaks_1_uA, cathodic_peaks_2_uA):
    # Drawing the Linear Regression fit line
    anodic_slope, anodic_intercept = np.polyfit(sqrt_scan_rates, anodic_peaks_uA, 1)
    cathodic_1_slope, cathodic_1_intercept = np.polyfit(sqrt_scan_rates, cathodic_peaks_1_uA, 1)
    cathodic_2_slope, cathodic_2_intercept = np.polyfit(sqrt_scan_rates, cathodic_peaks_2_uA, 1)

    # Calculate R^2 values
    anodic_r2 = np.corrcoef(sqrt_scan_rates, anodic_peaks_uA)[0, 1]**2
    cathodic_1_r2 = np.corrcoef(sqrt_scan_rates, cathodic_peaks_1_uA)[0, 1]**2
    cathodic_2_r2 = np.corrcoef(sqrt_scan_rates, cathodic_peaks_2_uA)[0, 1]**2

    anodic_fit = anodic_slope * np.array(sqrt_scan_rates) + anodic_intercept
    cathodic_1_fit = cathodic_1_slope * np.array(sqrt_scan_rates) + cathodic_1_intercept
    cathodic_2_fit = cathodic_2_slope * np.array(sqrt_scan_rates) + cathodic_2_intercept
    return anodic_fit, anodic_r2, cathodic_1_fit, cathodic_1_r2, cathodic_2_fit, cathodic_2_r2    

def get_peak_currents(cv_path, cycle_number=1):
    """
    """
    # This function calculates peaks for all cycles and saves a CSV.
    peak_data = raw_cv_support.save_peaks_cv(cv_path)

    # Check if the requested cycle exists in the DataFrame
    if cycle_number >= len(peak_data):
        print(f"Error: Cycle {cycle_number} not found in the data from {cv_path}")
        return None, None

    # Get the row for the specified cycle
    cycle_peaks = peak_data.iloc[cycle_number]
    # Get the anodic peak current
    anodic_peak = cycle_peaks.get('Anodic_I(A)', np.nan)
    cathodic_peak_1 = cycle_peaks.get('Cathodic_1_I(A)', np.nan)
    cathodic_peak_2 = cycle_peaks.get('Cathodic_2_I(A)', np.nan)

    return anodic_peak, cathodic_peak_1, cathodic_peak_2

def randles_sevnick(scan_rates_v, cv_paths):
    """
    Analyzes a series of CV files to generate data for a Randles-Sevcik plot.

    Args:
        scan_rates_v (list or np.array): A list of scan rates in V/s.
        cv_paths (list): A list of file paths for the corresponding CV data.

    Returns:
        dict: A dictionary containing x-axis (sqrt_scan_rates) and y-axis (peak currents) data.
    """
    if len(scan_rates_v) != len(cv_paths):
        raise ValueError("The number of scan rates must match the number of CV paths.")

    anodic_peaks = []
    cathodic_peaks_1 = []
    cathodic_peaks_2 = []

    for path in cv_paths:
        anodic, cathodic_1, cathodic_2 = get_peak_currents(path, cycle_number=0)
        anodic_peaks.append(anodic)
        cathodic_peaks_1.append(cathodic_1)
        cathodic_peaks_2.append(cathodic_2)
   
    anodic_peaks_uA = [p * 1e6 for p in anodic_peaks]
    cathodic_peaks_1_uA = [p * 1e6 for p in cathodic_peaks_1]
    cathodic_peaks_2_uA = [p * 1e6 for p in cathodic_peaks_2]
    
    return np.sqrt(scan_rates_v), np.array(anodic_peaks_uA), np.array(cathodic_peaks_1_uA), np.array(cathodic_peaks_2_uA )

