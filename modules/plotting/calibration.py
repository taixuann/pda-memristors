import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from modules.utils import FIGURES_DIR, RAW_DATA, save_json

def run_calibration(sce_path, pH):

    print("\n=== Running Calibration from Fc file ===")

    try:
        # Define the columns we expect to use. This makes parsing more robust
        # and ignores the problematic extra column caused by a trailing semicolon.
        expected_cols = [
            'Potential applied (V)', 'Time (s)', 'WE(1).Current (A)',
            'WE(1).Potential (V)', 'Scan', 'Index', 'Q+', 'Q-', 'Current range'
        ]
        data = pd.read_csv(
            sce_path,
            sep=';',
            decimal=',',
            header=0,
            usecols=expected_cols
        )
    except pd.errors.ParserError as e:
        print(f"Error parsing file {sce_path} with sep=';' and decimal=','.")
        print(f"Pandas Error: {e}")
        print("Please check the file to ensure it is correctly formatted. The error often indicates an incorrect number of columns on a specific line.")
        return  # Exit if parsing fails

    potential = data['WE(1).Potential (V)'].values
    current = data['WE(1).Current (A)'].values
    scans = data['Scan'].values
    uniq_scans = pd.unique(scans)

    cycles = [(np.where(scans == s)[0][0], np.where(scans == s)[0][-1]+1) for s in uniq_scans]
    num_cycles = len(cycles)

    E_half = []
    for (start, end) in cycles:
        E = potential[start:end]; I = current[start:end]
        E_half.append((E[np.argmax(I)] + E[np.argmin(I)]) / 2)

    E_half_mean_SCE = np.mean(E_half)
    E_Fc_NHE = 0.63
    E_sce_to_nhe = E_Fc_NHE - E_half_mean_SCE
    E_sce_to_rhe = E_sce_to_nhe + 0.059 * pH

    # Save json
    json_filename = f"RHE_{os.path.splitext(os.path.basename(sce_path))[0]}.json"
    results = {
        "E_half_mean_SCE": E_half_mean_SCE,
        "E_sce_to_rhe": E_sce_to_rhe,
        "E_sce_to_nhe": E_sce_to_nhe,
        "pH": pH
    }
    save_json(results, json_filename)

    #Save figure
    colors = plt.cm.plasma(np.linspace(0, 1, num_cycles))
    plt.figure(figsize=(12,6))
    for i, (start, end) in enumerate(cycles):
        E, I = potential[start:end], current[start:end]
        plt.plot(E, I, color=colors[i], lw=1.4, alpha=1.0, label=f"Cycle {i+1} SCE")
    plt.xlabel("Potential (V)")
    plt.ylabel("Current (A)")
    plt.title("Calibration with Fc", fontsize=16, weight='bold')
    plt.grid(True, linestyle = '--', alpha = 0.5)

    handles = [plt.Line2D([0],[0], color=colors[i], lw=2) for i in range(num_cycles)]
    labels = [f"Cycle {i+1}" for i in range(num_cycles)]
    if num_cycles < 15:
        plt.legend(handles, labels, loc='best', title="vs RHE")
        plt.tight_layout()
    elif 15 <= num_cycles <= 30:
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs SCE", ncol=1, fontsize=9)
        plt.tight_layout(rect=[0,0,0.80,1])
        plt.subplots_adjust(right=0.78)
    else:
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs SCE", ncol=2, fontsize=8)
        plt.tight_layout(rect=[0,0,0.72,1])
        plt.subplots_adjust(right=0.7)

    fig_path = FIGURES_DIR / f"Calibration_{os.path.splitext(os.path.basename(sce_path))[0]}.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()
    print(f"✔ Calibration plot saved to {fig_path}")

    return E_sce_to_rhe
