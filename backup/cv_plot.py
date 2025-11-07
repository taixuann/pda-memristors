#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import subprocess
from scipy.signal import find_peaks
import argparse

# === Configuration for parser linking ===

parser = argparse.ArgumentParser(description="CV Plot Automation Script")
parser.add_argument("--sce", type=str, required=True, help="Path to SCE calibration file")
parser.add_argument("--pretreat", type=str, required=True, help="Path to Pre-treatment CV file")
parser.add_argument("--cv", type=str, required=True, help="Path to PDA CV data file")
parser.add_argument("--pH", type=float, required=True, help="pH value for RHE conversion")
parser.add_argument("--tags-date", type=str, required=True, help="Date tag for Finder tagging")
parser.add_argument("--highlight-scans", type=lambda x: x.lower() == 'true', default=True, help="Set to 'true' to highlight first, second, third, and final scans.")
parser.add_argument("--highlight-colors", type=str, default="#1f77b4,#2ca02c,#d62728,#000000", help="Comma-separated list of 4 colors for highlighted scans (1st, 2nd, 3rd, final).")
args = parser.parse_args()

sce_calib_path = args.sce
pretreat_path = args.pretreat
cv_path = args.cv
pH = args.pH

# === Tag function ===

tags_calibration = (args.tags_date, "calibration")
tags_pre_treatment = (args.tags_date, "pre-treat")
tags_cv_sce = (args.tags_date, "SCE")
tags_cv_rhe = (args.tags_date, "RHE")
tags_peaks = (args.tags_date, "peaks")
tags_calibration_peaks = (args.tags_date, "peaks-calibration")

print(f"\n[Parser Loaded]")
print(f"SCE file: {sce_calib_path}")
print(f"Pre-treat file: {pretreat_path}")
print(f"CV file: {cv_path}")
print(f"pH: {pH}")
print(f"Tags date: {args.tags_date}\n")

# ===============================================================
# === 1. FILE PATH DIRECTORY ===
# ===============================================================

sce_calib_base = os.path.splitext(os.path.basename(sce_calib_path))[0]
pretreat_base = os.path.splitext(os.path.basename(pretreat_path))[0]
cv_base = os.path.splitext(os.path.basename(cv_path))[0]

output_dir_csv = "/Users/tai/Downloads/Research/PDA-based memristors/plotting/results/data"
os.makedirs(output_dir_csv, exist_ok=True)
output_dir_png = "/Users/tai/Downloads/Research/PDA-based memristors/plotting/results/figures"
os.makedirs(output_dir_png, exist_ok=True)
output_dir_sum = "/Users/tai/Downloads/Research/PDA-based memristors/plotting/results/sum"
os.makedirs(output_dir_png, exist_ok=True)

# ===============================================================
# === 2. CALIBRATION SECTION (SCE → RHE/NHE) ===
# ===============================================================
print("\n=== Step 1: Running Calibration from Fc file ===")

# Read input
data_calib = pd.read_csv(sce_calib_path, sep=';', decimal=',')
potential_calib = data_calib['WE(1).Potential (V)'].values
current_calib = data_calib['WE(1).Current (A)'].values

scans = data_calib["Scan"].values
uniq_scans = pd.unique(scans)
cycles = []
for s in uniq_scans:
    idxs = np.where(scans == s)[0]
    if idxs.size > 0:
        cycles.append((int(idxs[0]), int(idxs[-1]) + 1))

# Plotting calibration with defining peaks and peak-to-peak separation
E_half_list = []
plt.figure(figsize=(8,6))
for i, (start, end) in enumerate(cycles):
    E = potential_calib[start:end]
    I = current_calib[start:end]
    plt.plot(E, I, label=f"Cycle {i+1}")
    E_pa, E_pc = E[np.argmax(I)], E[np.argmin(I)]
    E_half = (E_pa + E_pc)/2
    E_half_list.append(E_half)

E_Fc_NHE_lit = 0.63

E_half_mean_SCE = np.mean(E_half_list) # This is the E_1/2 of Fc/Fc+ vs. SCE
E_sce_to_nhe = E_Fc_NHE_lit - E_half_mean_SCE
E_sce_to_rhe = E_sce_to_nhe + 0.059 * pH
E_half_mean_RHE = E_half_mean_SCE + E_sce_to_rhe
E_half_mean_NHE = E_half_mean_SCE + E_sce_to_nhe

calib_data = {
    "E_half_mean_SCE": round(float(E_half_mean_SCE), 4),
    "E_half_mean_RHE": round(float(E_half_mean_RHE), 4),
    "E_half_mean_NHE": round(float(E_half_mean_NHE), 4),
    "E_sce_to_rhe": round(float(E_sce_to_rhe), 4),
    "E_sce_to_nhe": round(float(E_sce_to_nhe), 4),
    "pH": pH
}
calib_file = os.path.join(output_dir_csv, f"RHE_{sce_calib_base}.json")
with open(calib_file, "w") as f:
    json.dump(calib_data, f, indent=4)

plt.title("SCE Calibration CV", fontsize=14, weight='bold')
plt.xlabel("Potential (V vs SCE)")
plt.ylabel("Current (A)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
calib_png_file = os.path.join(output_dir_png, f"{sce_calib_base}_plot.png")
plt.savefig(calib_png_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✔ Calibration done. Offset (SCE→RHE): {E_sce_to_rhe:.3f} V")
print(f"Calibration data saved to {calib_file}\n")

for t in tags_calibration:
    subprocess.run(["tag", "-a", t, calib_file])
for t in tags_calibration:
    subprocess.run(["tag", "-a", t, calib_png_file])

# ---------- Detect and print anodic/cathodic peaks for each cycle ----------

print("\n=== Step 1.2: Detecting and Printing Anodic/Cathodic Peaks for Each Cycle for SCE calibration ===")
anodic_peaks_calib = []
cathodic_peaks_calib = []
for idx, (start, end) in enumerate(cycles):
    E = potential_calib[start:end]
    I = current_calib[start:end]

    peaks_anodic, _ = find_peaks(I)
    peaks_cathodic, _ = find_peaks(-I)

    if len(peaks_anodic) > 0:
        max_peak_idx = peaks_anodic[np.argmax(I[peaks_anodic])]
        anodic_peaks_calib.append((float(E[max_peak_idx]), float(I[max_peak_idx])))
    else:
        anodic_peaks_calib.append((np.nan, np.nan))

    if len(peaks_cathodic) > 0:
        min_peak_idx = peaks_cathodic[np.argmin(I[peaks_cathodic])]
        cathodic_peaks_calib.append((float(E[min_peak_idx]), float(I[min_peak_idx])))
    else:
        cathodic_peaks_calib.append((np.nan, np.nan))

for i, (a_sce, c_sce) in enumerate(
    zip(anodic_peaks_calib, cathodic_peaks_calib)
):
    aE_calib, aI_calib = a_sce
    cE_calib, cI_calib = c_sce
    print(f"Cycle {i+1:02d}:")
    print(f"  Anodic (SCE):  E = {aE_calib:.2f} V, I = {aI_calib:.2e} A")
    print(f"  Cathodic (SCE): E = {cE_calib:.2f} V, I = {cI_calib:.2e} A")

peak_data = pd.DataFrame({
    'Cycle': [f'Cycle {i+1}' for i in range(len(cycles))],
    'Anodic_E_SCE(V)': [a[0] for a in anodic_peaks_calib],
    'Anodic_I_SCE(A)': [a[1] for a in anodic_peaks_calib],
    'Cathodic_E_SCE(V)': [c[0] for c in cathodic_peaks_calib],
    'Cathodic_I_SCE(A)': [c[1] for c in cathodic_peaks_calib]
})
peaks_calib_path = os.path.join(output_dir_csv, f"{sce_calib_base}_peaks.csv")
peak_data.to_csv(peaks_calib_path, index=False)
print(f"Peak data saved to {peaks_calib_path}")

for t in tags_calibration_peaks:
    subprocess.run(["tag", "-a", t, peaks_calib_path])

# ===============================================================
# === 3b. MAIN CV PLOT vs SCE ===
# ===============================================================
print("=== Step 3a: Plotting Main CV with SCE Conversion ===")

data_cv = pd.read_csv(cv_path, sep=';', decimal=',')
potential = data_cv['WE(1).Potential (V)'].values
current = data_cv['WE(1).Current (A)'].values

# detect cycles automatically (prefer explicit 'Scan' column if present)
cycles = []
# If data file contains a Scan column, use it to build cycles exactly
if 'Scan' in data_cv.columns or 'scan' in data_cv.columns:
    scan_col = 'Scan' if 'Scan' in data_cv.columns else 'scan'
    scans = data_cv[scan_col].values
    uniq_scans = pd.unique(scans)
    for s in uniq_scans:
        idxs = np.where(scans == s)[0]
        if idxs.size > 0:
            cycles.append((int(idxs[0]), int(idxs[-1]) + 1))
else:
    # fallback: detect cycles by potential reversals (original robust method)
    dV = np.diff(potential)
    direction = np.sign(dV)
    reversals = np.where(np.diff(direction) != 0)[0]
    reversals = [0] + reversals.tolist() + [len(potential)]
    cycles = [(reversals[i], reversals[i+1]) for i in range(len(reversals)-1)]

# continue with plotting
plt.figure(figsize=(12,6))
num_cycles = len(cycles)
highlight_indices = {0, 1, 2, num_cycles - 1}

if args.highlight_scans:
    # --- Generate Highlighted Plot First ---
    custom_colors = args.highlight_colors.split(',')
    if len(custom_colors) != 4:
        raise ValueError("Please provide exactly 4 colors for --highlight-colors.")
    highlight_colors = {
        0: custom_colors[0], 
        1: custom_colors[1], 
        2: custom_colors[2], 
        num_cycles - 1: custom_colors[3]}
    for i, (start, end) in enumerate(cycles):
        E, I = potential[start:end], current[start:end]
        if i in highlight_indices:
            color = highlight_colors.get(i, '#000000') # Default to black for final scan if it overlaps
            plt.plot(E, I, color=color, lw=1.8, alpha=1.0, label=f"Cycle {i+1} SCE")
        else:
            plt.plot(E, I, color='lightgray', lw=1.0, alpha=0.5, label=f"Cycle {i+1} SCE")

    plt.title("Cyclic Voltammetry - PDA (Highlighted)", fontsize=16, weight='bold')
    plt.xlabel("Potential (V vs SCE)", fontsize=14)
    plt.ylabel("Current (A)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    sce_handles = []
    for i in range(num_cycles):
        color = 'lightgray'
        if i in highlight_indices:
            color = highlight_colors.get(i)
            # Handle case where final scan is also one of the first three
            if i == num_cycles - 1:
                color = highlight_colors[num_cycles - 1]
        sce_handles.append(plt.Line2D([0], [0], color=color, lw=2))
    sce_labels = [f"Cycle {i+1}" for i in range(num_cycles)]

    # Legend placement logic for SCE plot
    if num_cycles < 15:
        plt.legend(sce_handles, sce_labels, loc='best', title="vs SCE")
        plt.tight_layout()
    elif 15 <= num_cycles <= 30:
        plt.legend(sce_handles, sce_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs SCE", ncol=1, fontsize=9, frameon=True)
        plt.tight_layout(rect=[0, 0, 0.80, 1])
        plt.subplots_adjust(right=0.78)
    else:
        plt.legend(sce_handles, sce_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs SCE", ncol=2, fontsize=8, frameon=True)
        plt.tight_layout(rect=[0, 0, 0.72, 1])
        plt.subplots_adjust(right=0.7)
    
    cv_sce_png_file_highlight = os.path.join(output_dir_png, f"{cv_base}_plot_SCE_highlight.png")
    plt.savefig(cv_sce_png_file_highlight, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Main CV plot (highlighted) saved to {os.path.basename(cv_sce_png_file_highlight)}")
    for t in tags_cv_sce:
        subprocess.run(["tag", "-a", t, cv_sce_png_file_highlight])

    # --- Now, create the standard plot ---
    plt.figure(figsize=(12,6))

# --- Generate Standard Plot (always) ---
colors = plt.cm.plasma(np.linspace(0, 1, len(cycles)))
for i, (start, end) in enumerate(cycles):
    E, I = potential[start:end], current[start:end]
    plt.plot(E, I, color=colors[i], lw=1.4, alpha=1.0, label=f"Cycle {i+1} SCE")

plt.title("Cyclic Voltammetry - PDA", fontsize=16, weight='bold')
plt.xlabel("Potential (V vs SCE)", fontsize=14)
plt.ylabel("Current (A)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

colors = plt.cm.plasma(np.linspace(0, 1, len(cycles)))
sce_handles = [plt.Line2D([0],[0],color=colors[i],lw=2) for i in range(len(cycles))]
sce_labels = [f"Cycle {i+1}" for i in range(len(cycles))]

# Legend placement logic for SCE plot
if num_cycles < 15:
    # Few cycles: legend inside plot, single column
    plt.legend(
        sce_handles, sce_labels,
        loc='best',
        title="vs SCE"
    )
    plt.tight_layout()
elif 15 <= num_cycles <= 30:
    # Moderate cycles: legend outside, single column
    plt.legend(
        sce_handles, sce_labels,
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        title="vs SCE", ncol=1, fontsize=9, frameon=True
    )
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.subplots_adjust(right=0.78)
else:
    # Many cycles: legend outside, two columns
    plt.legend(
        sce_handles, sce_labels,
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        title="vs SCE", ncol=2, fontsize=8, frameon=True
    )
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.subplots_adjust(right=0.7)
cv_sce_png_file = os.path.join(output_dir_png, f"{cv_base}_plot_SCE.png")
plt.savefig(cv_sce_png_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✔ Main CV plot saved to {os.path.basename(cv_sce_png_file)}\n")

for t in tags_cv_sce:
    subprocess.run(["tag", "-a", t, cv_sce_png_file])

# ===============================================================
# === 4. MAIN CV PLOT WITH RHE CONVERSION ===
# ===============================================================
print("=== Step 3b: Plotting Main CV with RHE Conversion ===")

data_cv = pd.read_csv(cv_path, sep=';', decimal=',')
potential = data_cv['WE(1).Potential (V)'].values
current = data_cv['WE(1).Current (A)'].values
potential_rhe = potential + E_sce_to_rhe

# detect cycles automatically (prefer explicit 'Scan' column if present)
cycles = []
# If data file contains a Scan column, use it to build cycles exactly
if 'Scan' in data_cv.columns or 'scan' in data_cv.columns:
    scan_col = 'Scan' if 'Scan' in data_cv.columns else 'scan'
    scans = data_cv[scan_col].values
    uniq_scans = pd.unique(scans)
    for s in uniq_scans:
        idxs = np.where(scans == s)[0]
        if idxs.size > 0:
            cycles.append((int(idxs[0]), int(idxs[-1]) + 1))
else:
    # fallback: detect cycles by potential reversals (original robust method)
    dV = np.diff(potential)
    direction = np.sign(dV)
    reversals = np.where(np.diff(direction) != 0)[0]
    reversals = [0] + reversals.tolist() + [len(potential)]
    cycles = [(reversals[i], reversals[i+1]) for i in range(len(reversals)-1)]

# continue with plotting
plt.figure(figsize=(12,6))
num_cycles = len(cycles)
highlight_indices = {0, 1, 2, num_cycles - 1}

if args.highlight_scans:
    # --- Generate Highlighted Plot First ---
    custom_colors = args.highlight_colors.split(',')
    if len(custom_colors) != 4:
        raise ValueError("Please provide exactly 4 colors for --highlight-colors.")
    highlight_colors = {
        0: custom_colors[0], 
        1: custom_colors[1], 
        2: custom_colors[2], 
        num_cycles - 1: custom_colors[3]}
    for i, (start, end) in enumerate(cycles):
        E, I = potential[start:end], current[start:end]
        E_rhe = E + E_sce_to_rhe
        if i in highlight_indices:
            color = highlight_colors.get(i, '#000000') # Default to black for final scan if it overlaps
            plt.plot(E_rhe, I, color=color, lw=1.8, alpha=1.0, label=f"Cycle {i+1} RHE")
        else:
            plt.plot(E_rhe, I, color='lightgray', lw=1.0, alpha=0.5, label=f"Cycle {i+1} RHE")

    plt.title("Cyclic Voltammetry - PDA (Highlighted)", fontsize=16, weight='bold')
    plt.xlabel("Potential (V vs RHE)", fontsize=14)
    plt.ylabel("Current (A)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    rhe_handles = []
    for i in range(num_cycles):
        color = 'lightgray'
        if i in highlight_indices:
            color = highlight_colors.get(i)
            # Handle case where final scan is also one of the first three
            if i == num_cycles - 1:
                color = highlight_colors[num_cycles - 1]
        rhe_handles.append(plt.Line2D([0], [0], color=color, lw=2))
    rhe_labels = [f"Cycle {i+1}" for i in range(num_cycles)]

    # Legend placement logic for RHE plot
    if num_cycles < 15:
        plt.legend(rhe_handles, rhe_labels, loc='best', title="vs RHE")
        plt.tight_layout()
    elif 15 <= num_cycles <= 30:
        plt.legend(rhe_handles, rhe_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs RHE", ncol=1, fontsize=9, frameon=True)
        plt.tight_layout(rect=[0, 0, 0.80, 1])
        plt.subplots_adjust(right=0.78)
    else:
        plt.legend(rhe_handles, rhe_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs RHE", ncol=2, fontsize=8, frameon=True)
        plt.tight_layout(rect=[0, 0, 0.72, 1])
        plt.subplots_adjust(right=0.7)
    
    cv_rhe_png_file_highlight = os.path.join(output_dir_png, f"{cv_base}_plot_RHE_highlight.png")
    plt.savefig(cv_rhe_png_file_highlight, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Main CV plot (highlighted) saved to {os.path.basename(cv_rhe_png_file_highlight)}")
    for t in tags_cv_rhe:
        subprocess.run(["tag", "-a", t, cv_rhe_png_file_highlight])

    # --- Now, create the standard plot ---
    plt.figure(figsize=(12,6))

# --- Generate Standard Plot (always) ---
    colors = plt.cm.plasma(np.linspace(0, 1, len(cycles)))
    for i, (start, end) in enumerate(cycles):
        E, I = potential[start:end], current[start:end]
        E_rhe = E + E_sce_to_rhe
        plt.plot(E_rhe, I, color=colors[i], lw=1.4, alpha=1.0, label=f"Cycle {i+1} RHE")

plt.title("Cyclic Voltammetry - PDA", fontsize=16, weight='bold')
plt.xlabel("Potential (V vs RHE)", fontsize=14)
plt.ylabel("Current (A)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

colors = plt.cm.plasma(np.linspace(0, 1, len(cycles)))
rhe_handles = [plt.Line2D([0],[0],color=colors[i],lw=2) for i in range(len(cycles))]
rhe_labels = [f"Cycle {i+1}" for i in range(len(cycles))]

# Legend placement logic for RHE plot
if num_cycles < 15:
    # Few cycles: legend inside plot, single column
    plt.legend(
        rhe_handles, rhe_labels,
        loc='best',
        title="vs RHE"
    )
    plt.tight_layout()
elif 15 <= num_cycles <= 30:
    # Moderate cycles: legend outside, single column
    plt.legend(
        rhe_handles, rhe_labels,
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        title="vs RHE", ncol=1, fontsize=9, frameon=True
    )
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.subplots_adjust(right=0.78)
else:
    # Many cycles: legend outside, two columns
    plt.legend(
        rhe_handles, rhe_labels,
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        title="vs RHE", ncol=2, fontsize=8, frameon=True
    )
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.subplots_adjust(right=0.7)
cv_rhe_png_file = os.path.join(output_dir_png, f"{cv_base}_plot_RHE.png")
plt.savefig(cv_rhe_png_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✔ Main CV plot saved to {os.path.basename(cv_rhe_png_file)}\n")

for t in tags_cv_rhe:
    subprocess.run(["tag", "-a", t, cv_rhe_png_file])

# ===============================================================
# === 5. Detect and Print Anodic/Cathodic Peaks for Each Cycle ===
# ===============================================================

print("=== Step 4: Detecting and Printing Anodic/Cathodic Peaks for Each Cycle ===")
anodic_peaks = []
cathodic_peaks = []
anodic_peaks_rhe = []
cathodic_peaks_rhe = []
for idx, (start, end) in enumerate(cycles):
    E = potential[start:end]
    I = current[start:end]
    E_rhe = E + E_sce_to_rhe
    if len(I) > 0:
        peaks_anodic, _ = find_peaks(I)
        peaks_cathodic, _ = find_peaks(-I)

        if len(peaks_anodic) > 0:
            max_peak_idx = peaks_anodic[np.argmax(I[peaks_anodic])]
            anodic_peaks.append((float(E[max_peak_idx]), float(I[max_peak_idx])))
            anodic_peaks_rhe.append((float(E_rhe[max_peak_idx]), float(I[max_peak_idx])))
        else:
            anodic_peaks.append((np.nan, np.nan))
            anodic_peaks_rhe.append((np.nan, np.nan))

        if len(peaks_cathodic) > 0:
            min_peak_idx = peaks_cathodic[np.argmin(I[peaks_cathodic])]
            cathodic_peaks.append((float(E[min_peak_idx]), float(I[min_peak_idx])))
            cathodic_peaks_rhe.append((float(E_rhe[min_peak_idx]), float(I[min_peak_idx])))
        else:
            cathodic_peaks.append((np.nan, np.nan))
            cathodic_peaks_rhe.append((np.nan, np.nan))
    else:
        anodic_peaks.append((np.nan, np.nan))
        cathodic_peaks.append((np.nan, np.nan))
        anodic_peaks_rhe.append((np.nan, np.nan))
        cathodic_peaks_rhe.append((np.nan, np.nan))

print("\n=== Peak summary (SCE and RHE) ===")
for i, (a_sce, c_sce, a_rhe, c_rhe) in enumerate(
    zip(anodic_peaks, cathodic_peaks, anodic_peaks_rhe, cathodic_peaks_rhe)
):
    aE_sce, aI_sce = a_sce
    cE_sce, cI_sce = c_sce
    aE_rhe, aI_rhe = a_rhe
    cE_rhe, cI_rhe = c_rhe
    print(f"Cycle {i+1:02d}:")
    print(f"  Anodic (SCE):  E = {aE_sce:.4f} V, I = {aI_sce:.4e} A")
    print(f"  Cathodic (SCE): E = {cE_sce:.4f} V, I = {cI_sce:.4e} A")
    print(f"  Anodic (RHE):  E = {aE_rhe:.4f} V, I = {aI_rhe:.4e} A")
    print(f"  Cathodic (RHE): E = {cE_rhe:.4f} V, I = {cI_rhe:.4e} A\n")

# Save peak data to CSV
peak_data = pd.DataFrame({
    'Cycle': [f'Cycle {i+1}' for i in range(len(cycles))],
    'Anodic_E_SCE(V)': [a[0] for a in anodic_peaks],
    'Anodic_I_SCE(A)': [a[1] for a in anodic_peaks],
    'Cathodic_E_SCE(V)': [c[0] for c in cathodic_peaks],
    'Cathodic_I_SCE(A)': [c[1] for c in cathodic_peaks],
    'Anodic_E_RHE(V)': [a[0] for a in anodic_peaks_rhe],
    'Anodic_I_RHE(A)': [a[1] for a in anodic_peaks_rhe],
    'Cathodic_E_RHE(V)': [c[0] for c in cathodic_peaks_rhe],
    'Cathodic_I_RHE(A)': [c[1] for c in cathodic_peaks_rhe]
})
peaks_csv_path = os.path.join(output_dir_csv, f"{cv_base}_peaks.csv")
peak_data.to_csv(peaks_csv_path, index=False)
print(f"Peak data saved to {peaks_csv_path}")

for t in tags_peaks:
    subprocess.run(["tag", "-a", t, peaks_csv_path])

# ===============================================================
# === 3. PRE-TREAT CV PLOT ===
# ===============================================================
print("=== Step 2: Plotting Pre-treatment CV ===")

data_pre = pd.read_csv(pretreat_path, sep=';', decimal=',')
possible_cols = ['Cycle', 'cycle', 'Scan', 'scan', 'CycleIndex', 'Scan #', 'ScanNumber']
cycle_col = next((c for c in possible_cols if c in data_pre.columns), None)
if not cycle_col:
    raise ValueError("No cycle column found in pre-treat file!")

potential = data_pre['WE(1).Potential (V)']
current = data_pre['WE(1).Current (A)']
cycles = data_pre[cycle_col].astype(int).values
unique_cycles = sorted(data_pre[cycle_col].unique())

plt.figure(figsize=(12,6))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cycles)))
for i, cyc in enumerate(unique_cycles):
    mask = (cycles == cyc)
    plt.plot(potential[mask], current[mask], color=colors[i], linewidth=1.6, label=f"Cycle {cyc}")

plt.title("Pre-treatment CV in PBS", fontsize=16, weight='bold')
plt.xlabel("Potential (V vs SCE)", fontsize=14)
plt.ylabel("Current (A)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
x_min, x_max = potential.min(), potential.max()
padding = 0.1 * (x_max - x_min)
plt.xlim(x_min - padding, x_max + padding)
plt.legend(title="Cycles", loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.tight_layout(rect=[0,0,0.8,1])
pretreat_png_file = os.path.join(output_dir_png, f"{pretreat_base}_plot.png")
plt.savefig(pretreat_png_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✔ Pre-treat plot saved.\n")

for t in tags_pre_treatment:
    subprocess.run(["tag", "-a", t, pretreat_png_file])

print("🎯 All steps completed successfully!")