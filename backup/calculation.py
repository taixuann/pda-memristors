#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import argparse
import json

parser = argparse.ArgumentParser(description="Peak Calculation Script")
parser.add_argument("--peaks", type=str, required=True, help="Path to peaks CSV file (required)")
parser.add_argument("--tags-date", type=str, required=True, help="Date tag for Finder tagging (required)")
parser.add_argument("--calib-json", type=str, help="Path to RHE calibration JSON file")
args = parser.parse_args()

# Load potential shift from calibration JSON if provided
if args.calib_json and os.path.exists(args.calib_json):
    with open(args.calib_json, "r") as f:
        calib_data = json.load(f)
    E_sce_to_rhe = calib_data.get("E_sce_to_rhe", 0.7)
    print(f"[Auto] Loaded potential shift (SCE→RHE): {E_sce_to_rhe:.3f} V from JSON")
else:
    E_sce_to_rhe = 0.7
    print(f"[Warning] Calibration JSON not found, using default shift = {E_sce_to_rhe:.3f} V")

peaks_calib_path = args.peaks
tags = ["peak-analysis", args.tags_date]

print(f"\n[Parser Loaded]")
print(f"Peak file: {peaks_calib_path}")
print(f"Tags: {tags}\n")

# ===============================================================
# === 1. USER INPUT ===
# ===============================================================
peaks_cv_base = os.path.splitext(os.path.basename(peaks_calib_path))[0]
output_dir_csv = "/Users/tai/Downloads/Research/PDA-based memristors/plotting/data"
os.makedirs(output_dir_csv, exist_ok=True)
output_dir_png = "/Users/tai/Downloads/Research/PDA-based memristors/plotting/graphics"
os.makedirs(output_dir_csv, exist_ok=True)

# ===============================================================
# === 2. LOAD DATA ===
# ===============================================================
df = pd.read_csv(peaks_calib_path)

# Choose RHE or SCE columns
E_pa = df["Anodic_E_SCE(V)"]
I_pa = df["Anodic_I_SCE(A)"] * 1e6  # Convert A to µA
E_pc = df["Cathodic_E_SCE(V)"]
I_pc = df["Cathodic_I_SCE(A)"] * 1e6  # Convert A to µA

# ===============================================================
# === 3. COMPUTE ELECTROCHEMICAL PARAMETERS ===
# ===============================================================

df["Anodic_I_SCE(µA)"] = I_pa
df["Cathodic_I_SCE(µA)"] = I_pc

df["ΔE_p(V)"] = (E_pa - E_pc).abs()
df["i_ox/i_red"] = np.abs(I_pa) / np.abs(I_pc)
df["ΔI(µA)"] = I_pa - I_pc
df["n_electron_estimated"] = 0.059 / df["ΔE_p(V)"]  # at 25°C (V units)

# ===============================================================
# === 5. PRINT SUMMARY & ADD AVERAGE ROW ===
# ===============================================================

# Compute the mean for each numeric column of interest
mean_values = {
    "Anodic_E_SCE(V)": df["Anodic_E_SCE(V)"].mean(),
    "Anodic_I_SCE(µA)": df["Anodic_I_SCE(µA)"].mean(),
    "Cathodic_E_SCE(V)": df["Cathodic_E_SCE(V)"].mean(),
    "Cathodic_I_SCE(µA)": df["Cathodic_I_SCE(µA)"].mean(),
    "ΔE_p(V)": df["ΔE_p(V)"].mean(),
    "i_ox/i_red": df["i_ox/i_red"].mean(),
    "ΔI(µA)": df["ΔI(µA)"].mean(),
    "n_electron_estimated": df["n_electron_estimated"].mean()
}

# Prepare an "Average" row
average_row = {col: "" for col in df.columns}
for k, v in mean_values.items():
    average_row[k] = round(v, 3)
# Set the first column (if it's a cycle index or similar) to "Average" or blank
if df.columns[0] not in mean_values:
    average_row[df.columns[0]] = "Average"

# Append the average row to the DataFrame for pretty printing and export
df_with_avg = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)

# Round numeric outputs to 3 decimal places for printing and plotting
df_with_avg = df_with_avg.applymap(lambda x: round(x, 3) if isinstance(x, (int, float)) else x)

# Save extended DataFrame to new CSV in output_dir_png
analysis_avg_path = os.path.join(output_dir_png, f"{peaks_cv_base}_analysis_with_average.csv")
df_with_avg.to_csv(analysis_avg_path, index=False)

# Print a clean, table-like output for PowerPoint
print("=== Peak Analysis Table (including average) ===")
print(df_with_avg.to_string(index=False))

for t in tags:
    subprocess.run(["tag", "-a", t, analysis_avg_path])
print(f"✅ Added Finder tag '{tags}' to {peaks_calib_path}")
print(f"\n✔ Analysis results saved to: {analysis_avg_path}")
print(f"✔ Analysis table with average saved to: {analysis_avg_path}")

# ===============================================================
# === 6. PLOT RESULTS (DISPLAY ONLY, NOT SAVE) ===
# ===============================================================
# Select columns to show (all columns, or choose a subset)
columns_to_show = [
    "Cycle", "Anodic_E_SCE(V)", "Anodic_I_SCE(µA)", "Cathodic_E_SCE(V)", "Cathodic_I_SCE(µA)",
    "ΔE_p(V)", "i_ox/i_red", "ΔI(µA)", "n_electron_estimated"
]

# Create a figure for the table
fig_table, ax_table = plt.subplots(figsize=(12, len(df_with_avg) * 0.4))
ax_table.axis("off")  # Hide axes

# Create the table
table_data = df_with_avg[columns_to_show].values.tolist()
column_labels = df_with_avg[columns_to_show].columns
table = ax_table.table(
    cellText=table_data,
    colLabels=column_labels,
    loc="center",
    cellLoc="center"
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.2)
for key, cell in table.get_celld().items():
    if key[0] == 0:  # header row
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
    elif "Average" in str(cell.get_text().get_text()):
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight='bold')

plt.title("Peak Analysis Table (including average)", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()