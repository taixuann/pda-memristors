#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

# ===============================================================
# === 1. CONFIGURATION: Define experiment files and scan rates ===
# ===============================================================

# --- List of experiments to plot ---
# Each item is a dictionary containing the file path and the scan rate label.
EXPERIMENTS = [
    {
        "path": "/Users/tai/Downloads/Research/PDA-based memristors/plotting/raw_data/Fc_SCE_{Au (4), Pt small, SCE (Fc label)}_{sc25}_{29-10-25}.txt",
        "scan_rate": "25 mV/s"
    },
    {
        "path": "/Users/tai/Downloads/Research/PDA-based memristors/plotting/raw_data/Fc_SCE_{Au (4), Pt small, SCE (Fc label)}_{sc75}_{29-10-25}.txt",
        "scan_rate": "75 mV/s"
    },
    {
        "path": "/Users/tai/Downloads/Research/PDA-based memristors/plotting/raw_data/Fc_SCE_{Au (4), Pt small, SCE (Fc label)}_{sc100}_{29-10-25}.txt",
        "scan_rate": "100 mV/s"
    },
    {
        "path": "/Users/tai/Downloads/Research/PDA-based memristors/plotting/raw_data/Fc_SCE_{Au (4), Pt small, SCE (Fc label)}_{29-10-25}.txt",
        "scan_rate": "50 mV/s"
    },
    # Add more experiments here, for example:
    # {
    #     "path": "/path/to/your/50mV_data.txt",
    #     "scan_rate": "50 mV/s"
    # },
]

output_dir = "/Users/tai/Downloads/Research/PDA-based memristors/plotting/results/figures"
combined_output_filename = "summary_plot_combined_cv_and_regression.png"
os.makedirs(output_dir, exist_ok=True)

# --- Plotting Parameters ---
INTERPOLATION_POINTS = 500  # Number of points for the common potential axis

# ===============================================================
# === 2. HELPER FUNCTION: Get a specific cycle from a file ===
# ===============================================================

def get_cycle_data(file_path, cycle_number=2):
    """
    Loads a CV data file and extracts the data for a complete, full cycle.
    A full cycle consists of a forward and a backward scan.
    """
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: File not found, skipping: {os.path.basename(file_path)}")
        return None, None

    data = pd.read_csv(file_path, sep=';', decimal=',')
    
    # Use 'Scan' column if available for robust cycle detection
    scan_col = next((c for c in ['Scan', 'scan'] if c in data.columns), None)
    if scan_col:
        scans = data[scan_col].unique()
        if cycle_number > len(scans):
            print(f"⚠️  Warning: Cycle {cycle_number} not found in {os.path.basename(file_path)}. It only has {len(scans)} cycles.")
            return None, None
        target_scan = scans[cycle_number - 1]
        cycle_df = data[data[scan_col] == target_scan]
        return cycle_df['WE(1).Potential (V)'].values, cycle_df['WE(1).Current (A)'].values

    # Fallback to potential reversal detection
    full_potential = data['WE(1).Potential (V)'].values
    full_current = data['WE(1).Current (A)'].values
    reversals = np.where(np.diff(np.sign(np.diff(full_potential))) != 0)[0] + 1
    segments = np.split(np.arange(len(full_potential)), reversals)

    # A full cycle consists of two segments (e.g., forward and backward scan)
    start_segment_idx = (cycle_number - 1) * 2
    if start_segment_idx + 1 >= len(segments):
        print(f"⚠️  Warning: Cycle {cycle_number} not found in {os.path.basename(file_path)}. It only has {len(segments)//2} full cycles.")
        return None, None

    # Combine two consecutive segments to form a full cycle
    full_cycle_indices = np.concatenate((segments[start_segment_idx], segments[start_segment_idx + 1]))
    
    cycle_potential = full_potential[full_cycle_indices]
    cycle_current = full_current[full_cycle_indices]

    return cycle_potential, cycle_current

# ===============================================================
# === 3. MAIN SCRIPT: Generate and save the summary plot ===
# ===============================================================

# --- Create a figure with two subplots: one for CVs, one for regression ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
plt.style.use('seaborn-v0_8-whitegrid')

anodic_peaks = []
cathodic_peaks = []
# --- New lists to store data for regression plot ---
scan_rates_sqrt = []
anodic_peak_currents = []
cathodic_peak_currents = []

for exp in EXPERIMENTS:
    # Get data for the 2nd cycle
    potential, current = get_cycle_data(exp["path"], cycle_number=2)
    
    if potential is not None and current is not None:
        # Plot the full CV curve for the cycle and get the line color
        line, = ax1.plot(potential, current, label=f"{exp['scan_rate']}")
        
        # Find and store anodic and cathodic peaks
        anodic_peak_idx = np.argmax(current)
        cathodic_peak_idx = np.argmin(current)
        
        anodic_peaks.append((potential[anodic_peak_idx], current[anodic_peak_idx]))
        cathodic_peaks.append((potential[cathodic_peak_idx], current[cathodic_peak_idx]))

        # --- Store data for the regression plot ---
        scan_rate_val = float(exp['scan_rate'].split()[0]) # Extract numeric value
        scan_rates_sqrt.append(np.sqrt(scan_rate_val))
        anodic_peak_currents.append(current[anodic_peak_idx])
        cathodic_peak_currents.append(current[cathodic_peak_idx])

# --- Sort peaks by potential to ensure lines are drawn correctly ---
anodic_peaks.sort(key=lambda p: p[0])
cathodic_peaks.sort(key=lambda p: p[0])

# --- Plot the lines connecting the peaks ---
if anodic_peaks:
    ax1.plot([p[0] for p in anodic_peaks], [p[1] for p in anodic_peaks], 'r-o', lw=2, label='Anodic Peaks Trend')
if cathodic_peaks:
    ax1.plot([p[0] for p in cathodic_peaks], [p[1] for p in cathodic_peaks], 'b-o', lw=2, label='Cathodic Peaks Trend')

ax1.set_title("A) Cyclic Voltammograms (Cycle 2)", fontsize=16, weight='bold')
ax1.set_xlabel("Potential (V vs SCE)", fontsize=12)
ax1.set_ylabel("Current (A)", fontsize=12)
ax1.legend(title="Scan Rate", loc='best')
ax1.axhline(0, color='black', linewidth=0.5) # x-axis line
ax1.axvline(0, color='black', linewidth=0.5) # y-axis line

# ===============================================================
# === 4. CREATE AND SAVE REGRESSION PLOT (i vs. sqrt(scan rate)) ===
# ===============================================================

if scan_rates_sqrt:
    # --- Anodic Peak Regression ---
    # Convert to numpy arrays for calculations
    x_vals = np.array(scan_rates_sqrt)
    y_anodic = np.array(anodic_peak_currents)
    
    # Perform linear regression to get slope, intercept, and R-squared
    slope_anodic, intercept_anodic, r_value_anodic, _, _ = linregress(x_vals, y_anodic)
    r_squared_anodic = r_value_anodic**2
    
    ax2.plot(x_vals, y_anodic, 'ro', label='Anodic Peak Current (Ipa)')
    fit_label_anodic = f'Anodic Fit (R²={r_squared_anodic:.3f})'
    ax2.plot(x_vals, slope_anodic * x_vals + intercept_anodic, 'r--', label=fit_label_anodic)

    # --- Cathodic Peak Regression ---
    y_cathodic = np.array(cathodic_peak_currents)
    slope_cathodic, intercept_cathodic, r_value_cathodic, _, _ = linregress(x_vals, y_cathodic)
    r_squared_cathodic = r_value_cathodic**2

    ax2.plot(x_vals, y_cathodic, 'bo', label='Cathodic Peak Current (Ipc)')
    fit_label_cathodic = f'Cathodic Fit (R²={r_squared_cathodic:.3f})'
    ax2.plot(x_vals, slope_cathodic * x_vals + intercept_cathodic, 'b--', label=fit_label_cathodic)

    ax2.set_title("B) Peak Current vs. Sqrt(Scan Rate)", fontsize=16, weight='bold')
    ax2.set_xlabel("Scan Rate$^{1/2}$ (mV/s)$^{1/2}$", fontsize=12)
    ax2.set_ylabel("Peak Current (A)", fontsize=12)
    ax2.legend()
    ax2.grid(True)

fig.suptitle("Ferrocene Calibration Summary", fontsize=20, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle

output_path = os.path.join(output_dir, combined_output_filename)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ Combined summary plot saved successfully to: {output_path}")
