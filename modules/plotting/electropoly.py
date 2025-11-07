import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from modules.utils import FIGURES_DIR, DATA_DIR, RAW_DATA, read_json

def run_electropolymerization_SCE(pda_path, highlight_scans=True, highlight_colors="#1f77b4,#2ca02c,#d62728,#000000"):
    """
    Plot cyclic voltammetry for electropolymerization PDA data.
    
    Parameters:
    - pda_path: CSV filename in RAW_DATA folder
    - tag_date: experiment date tag (used for labeling)
    - highlight_scans: bool, whether to highlight selected cycles
    - highlight_colors: list of 4 color strings for highlighting (first 3 + last cycle)
    """
    # Build full path to raw data
    try:
        # Define the columns we expect to use. This makes parsing more robust
        # and ignores the problematic extra column caused by a trailing semicolon.
        expected_cols = [
            'Potential applied (V)', 'Time (s)', 'WE(1).Current (A)',
            'WE(1).Potential (V)', 'Scan', 'Index', 'Q+', 'Q-', 'Current range'
        ]
        data = pd.read_csv(
            pda_path,
            sep=';',
            decimal=',',
            header=0,
            usecols=expected_cols
        )
    except pd.errors.ParserError as e:
        print(f"Error parsing file {pda_path} with sep=';' and decimal=','.")
        print(f"Pandas Error: {e}")
        print("Please check the file to ensure it is correctly formatted. The error often indicates an incorrect number of columns on a specific line.")
        return  # Exit if parsing fails

    potential = data['WE(1).Potential (V)'].values
    current = data['WE(1).Current (A)'].values
    
    # --- Robust cycle detection ---
    cycles = []
    scan_col = next((c for c in ['Scan', 'scan'] if c in data.columns), None)
    if scan_col:
        print("Found 'Scan' column. Detecting cycles using scan numbers.")
        scans = data[scan_col].values
        uniq_scans = pd.unique(scans)
        for s in uniq_scans:
            idxs = np.where(scans == s)[0]
            if idxs.size > 0:
                cycles.append((int(idxs[0]), int(idxs[-1]) + 1))
    else:
        # Fallback: detect cycles by finding reversals in potential
        print("No 'Scan' column found. Detecting cycles by potential reversals.")
        dV = np.diff(potential)
        direction = np.sign(dV)
        reversals = np.where(np.diff(direction) != 0)[0]
        reversals = [0] + reversals.tolist() + [len(potential)]
        cycles = [(reversals[i], reversals[i+1]) for i in range(len(reversals)-1)]
    
    num_cycles = len(cycles)
    highlight_indices = {0, 1, 2, num_cycles - 1}

    # --- Highlighted plot (optional) ---
    if highlight_scans and highlight_colors:
        # Split the comma-separated string into a list of colors
        colors_list = highlight_colors.split(',')
        if len(colors_list) != 4:
            raise ValueError("highlight_colors must contain exactly 4 colors for first 3 + last cycles")
        color_map = { # Use the split list for mapping
            0: colors_list[0],
            1: colors_list[1],
            2: colors_list[2],
            num_cycles - 1: colors_list[3]
        }
        plt.figure(figsize=(12,6))
        for i, (start, end) in enumerate(cycles):
            E, I = potential[start:end], current[start:end]
            color = color_map.get(i, 'lightgray') if i in highlight_indices else 'lightgray'
            lw = 1.8 if i in highlight_indices else 1.0
            alpha = 1.0 if i in highlight_indices else 0.5
            plt.plot(E, I, color=color, lw=lw, alpha=alpha, label=f"Cycle {i+1} SCE")

        plt.title("Cyclic Voltammetry of deposition PDA (Highlighted)", fontsize=16, weight='bold')
        plt.xlabel("Potential (V vs SCE)", fontsize=14)
        plt.ylabel("Current (A)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Legend
        handles = [plt.Line2D([0],[0], color=color_map.get(i, 'lightgray'), lw=2) for i in range(num_cycles)]
        labels = [f"Cycle {i+1}" for i in range(num_cycles)]
        if num_cycles < 15:
            plt.legend(handles, labels, loc='best', title="vs SCE")
            plt.tight_layout()
        elif 15 <= num_cycles <= 30:
            plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs SCE", ncol=1, fontsize=9)
            plt.tight_layout(rect=[0,0,0.80,1])
            plt.subplots_adjust(right=0.78)
        else:
            plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs SCE", ncol=2, fontsize=8)
            plt.tight_layout(rect=[0,0,0.72,1])
            plt.subplots_adjust(right=0.7)

        fig_file_highlight = FIGURES_DIR / f"{Path(pda_path).stem}_plot_SCE_highlight.png"
        plt.savefig(fig_file_highlight, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✔ Highlighted CV plot saved to {fig_file_highlight.name}")

    # --- Standard plot ---
    plt.figure(figsize=(12,6))
    colors = plt.cm.plasma(np.linspace(0, 1, num_cycles))
    for i, (start, end) in enumerate(cycles):
        E, I = potential[start:end], current[start:end]
        plt.plot(E, I, color=colors[i], lw=1.4, alpha=1.0, label=f"Cycle {i+1} SCE")

    plt.title("Cyclic Voltammetry of deposition PDA", fontsize=16, weight='bold')
    plt.xlabel("Potential (V vs SCE)", fontsize=14)
    plt.ylabel("Current (A)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    handles = [plt.Line2D([0],[0], color=colors[i], lw=2) for i in range(num_cycles)]
    labels = [f"Cycle {i+1}" for i in range(num_cycles)]
    if num_cycles < 15:
        plt.legend(handles, labels, loc='best', title="vs SCE")
        plt.tight_layout()
    elif 15 <= num_cycles <= 30:
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs SCE", ncol=1, fontsize=9)
        plt.tight_layout(rect=[0,0,0.80,1])
        plt.subplots_adjust(right=0.78)
    else:
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs SCE", ncol=2, fontsize=8)
        plt.tight_layout(rect=[0,0,0.72,1])
        plt.subplots_adjust(right=0.7)

    fig_file = FIGURES_DIR / f"{Path(pda_path).stem}_plot_SCE.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✔ Standard CV plot saved to {fig_file.name}\n")

def run_electropolymerization_RHE(pda_path, highlight_scans=True, highlight_colors="#1f77b4,#2ca02c,#d62728,#000000"):
    # Build full path to raw data
    pda_path = RAW_DATA / pda_path
    try:
        # Define the columns we expect to use. This makes parsing more robust
        # and ignores the problematic extra column caused by a trailing semicolon.
        expected_cols = [
            'Potential applied (V)', 'Time (s)', 'WE(1).Current (A)',
            'WE(1).Potential (V)', 'Scan', 'Index', 'Q+', 'Q-', 'Current range'
        ]
        data = pd.read_csv(
            pda_path,
            sep=';',
            decimal=',',
            header=0,
            usecols=expected_cols
        )
    except pd.errors.ParserError as e:
        print(f"Error parsing file {pda_path} with sep=';' and decimal=','.")
        print(f"Pandas Error: {e}")
        print("Please check the file to ensure it is correctly formatted. The error often indicates an incorrect number of columns on a specific line.")
        return  # Exit if parsing fails

    potential = data['WE(1).Potential (V)'].values
    current = data['WE(1).Current (A)'].values
    
    # --- Robust cycle detection ---
    cycles = []
    scan_col = next((c for c in ['Scan', 'scan'] if c in data.columns), None)
    if scan_col:
        print("Found 'Scan' column. Detecting cycles using scan numbers.")
        scans = data[scan_col].values
        uniq_scans = pd.unique(scans)
        for s in uniq_scans:
            idxs = np.where(scans == s)[0]
            if idxs.size > 0:
                cycles.append((int(idxs[0]), int(idxs[-1]) + 1))
    else:
        # Fallback: detect cycles by finding reversals in potential
        print("No 'Scan' column found. Detecting cycles by potential reversals.")
        dV = np.diff(potential)
        direction = np.sign(dV)
        reversals = np.where(np.diff(direction) != 0)[0]
        reversals = [0] + reversals.tolist() + [len(potential)]
        cycles = [(reversals[i], reversals[i+1]) for i in range(len(reversals)-1)]
    
    num_cycles = len(cycles)
    highlight_indices = {0, 1, 2, num_cycles - 1}

    # --- Highlighted plot (optional) ---
    if highlight_scans and highlight_colors:
        # Split the comma-separated string into a list of colors
        colors_list = highlight_colors.split(',')
        if len(colors_list) != 4:
            raise ValueError("highlight_colors must contain exactly 4 colors for first 3 + last cycles")
        color_map = { # Use the split list for mapping
            0: colors_list[0],
            1: colors_list[1],
            2: colors_list[2],
            num_cycles - 1: colors_list[3]
        }
        plt.figure(figsize=(12,6))
        for i, (start, end) in enumerate(cycles):
            E, I = potential[start:end], current[start:end]
            color = color_map.get(i, 'lightgray') if i in highlight_indices else 'lightgray'
            lw = 1.8 if i in highlight_indices else 1.0
            alpha = 1.0 if i in highlight_indices else 0.5
            plt.plot(E, I, color=color, lw=lw, alpha=alpha, label=f"Cycle {i+1} RHE")

        plt.title("Cyclic Voltammetry of deposition PDA (Highlighted)", fontsize=16, weight='bold')
        plt.xlabel("Potential (V vs RHE)", fontsize=14)
        plt.ylabel("Current (A)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Legend
        handles = [plt.Line2D([0],[0], color=color_map.get(i, 'lightgray'), lw=2) for i in range(num_cycles)]
        labels = [f"Cycle {i+1}" for i in range(num_cycles)]
        if num_cycles < 15:
            plt.legend(handles, labels, loc='best', title="vs RHE")
            plt.tight_layout()
        elif 15 <= num_cycles <= 30:
            plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs RHE", ncol=1, fontsize=9)
            plt.tight_layout(rect=[0,0,0.80,1])
            plt.subplots_adjust(right=0.78)
        else:
            plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs RHE", ncol=2, fontsize=8)
            plt.tight_layout(rect=[0,0,0.72,1])
            plt.subplots_adjust(right=0.7)

        fig_file_highlight = FIGURES_DIR / f"{Path(pda_path).stem}_plot_RHE_highlight.png"
        plt.savefig(fig_file_highlight, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✔ Highlighted CV plot saved to {fig_file_highlight.name}")

    # --- Standard plot ---
    plt.figure(figsize=(12,6))
    colors = plt.cm.plasma(np.linspace(0, 1, num_cycles))
    for i, (start, end) in enumerate(cycles):
        E, I = potential[start:end], current[start:end]
        plt.plot(E, I, color=colors[i], lw=1.4, alpha=1.0, label=f"Cycle {i+1} SCE")

    plt.title("Cyclic Voltammetry of deposition PDA", fontsize=16, weight='bold')
    plt.xlabel("Potential (V vs RHE)", fontsize=14)
    plt.ylabel("Current (A)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    handles = [plt.Line2D([0],[0], color=colors[i], lw=2) for i in range(num_cycles)]
    labels = [f"Cycle {i+1}" for i in range(num_cycles)]
    if num_cycles < 15:
        plt.legend(handles, labels, loc='best', title="vs RHE")
        plt.tight_layout()
    elif 15 <= num_cycles <= 30:
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs RHE", ncol=1, fontsize=9)
        plt.tight_layout(rect=[0,0,0.80,1])
        plt.subplots_adjust(right=0.78)
    else:
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="vs RHE", ncol=2, fontsize=8)
        plt.tight_layout(rect=[0,0,0.72,1])
        plt.subplots_adjust(right=0.7)

    fig_file = FIGURES_DIR / f"{Path(pda_path).stem}_plot_RHE.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✔ Standard CV plot saved to {fig_file.name}\n")