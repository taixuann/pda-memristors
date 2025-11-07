#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import subprocess
from scipy.signal import find_peaks
import argparse

tags_calibration = ("31-10-25", "calibration")
tags_pre_treatment = ("31-10-25", "pre-treat")
tags_cv_sce = ("31-10-25", "SCE")
tags_cv_rhe = ("31-10-25", "RHE")
tags_peaks = ("31-10-25", "peaks")
tags_calibration_peaks = ("31-10-25", "peaks-calibration")
output_dir_csv = "/Users/tai/Downloads/Research/PDA-based memristors/plotting/data"
os.makedirs(output_dir_csv, exist_ok=True)
output_dir_png = "/Users/tai/Downloads/Research/PDA-based memristors/plotting/graphics"
os.makedirs(output_dir_png, exist_ok=True)
highlight_scans = True
highlight_colors = "#1f77b4,#2ca02c,#d62728,#000000"

print("=== Step 3a: Plotting Main CV with SCE Conversion ===")

data_cv = pd.read_csv("/Users/tai/Downloads/Research/PDA-based memristors/plotting/data/PDA_{1}_{GC (1), Pt small, SCE}_{10mV}_{31-10-25}.txt", sep=';', decimal=',')
cv_path = "/Users/tai/Downloads/Research/PDA-based memristors/plotting/data/PDA_{1}_{GC (1), Pt small, SCE}_{10mV}_{31-10-25}.txt"
cv_base = os.path.splitext(os.path.basename(cv_path))[0]
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


if highlight_scans:
    # --- Generate Highlighted Plot First ---
    custom_colors = highlight_colors.split(',')
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


 

 
