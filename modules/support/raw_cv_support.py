import matplotlib.pyplot as plt, numpy as np, pandas as pd
import os
from pathlib import Path
from modules import utils
from scipy.signal import find_peaks

def _create_plot(potential, current, cycles, num_cycles, title, xlabel, ylabel, fig_path, figsize=(12, 6), dpi=300, show_plot=True, close_plot=False, label_prefix="Cycle"):
    """Helper function to create and save a CV plot."""
    plt.figure(figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0, 1, num_cycles))
    for i, (start, end) in enumerate(cycles):
        E, I = potential[start:end], current[start:end]
        plt.plot(E, I, color=colors[i], lw=1.4, alpha=1.0, label=f"{label_prefix} {i+1}")
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    #plt.legend(loc = 'best')
    plt.tight_layout()
    
    os.makedirs(Path(fig_path).parent, exist_ok=True)
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    if close_plot:
        plt.close()

def plot_highlight(pda_path):
    """
    Plotting the highlight scans from the CV data.
    """
    # --- Color and splitting ---
    highlight_colors="#1f77b4,#2ca02c,#d62728,#000000"
    colors_list = highlight_colors.split(',')
    if len(colors_list) != 4:
        raise ValueError("highlight_colors must contain exactly 4 colors for first 3 + last cycles")
    
    highlight_indices = {0, 1, 2, num_cycles - 1}
    color_map = {idx: color for idx, color in zip(sorted(list(highlight_indices)), colors_list)}

    # --- Plotting ---
    potential, current, cycles, num_cycles = utils.read_data_cv(pda_path)
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
    labels = [f"Cycle {i+1}" for i in range(num_cycles)]
    plt.legend(labels=labels, loc = 'best')
    plt.tight_layout()

    # --- Save file ---
    fig_file_highlight = utils.FIGURES_DIR / f"{Path(pda_path).stem}_plot_SCE_highlight.png"
    plt.savefig(fig_file_highlight, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✔ Highlighted CV plot saved to {fig_file_highlight.name}")

def save_peaks_cv(cv_path):
    """
    Extract the peaks from the CV data
    """
    # --- Read CV data ---
    potential, current, cycles, num_cycles = utils.read_data_cv(cv_path)
    # --- Find peaks for each cycle ---
    all_cycle_peaks = []

    # Define potential windows for peak searching [min_V, max_V]
    # Based on your description:
    anodic_window = [0.2, 0.4]
    cathodic_window_1 = [-0.4, -0.2] # Peak between -0.4V and -0.2V
    cathodic_window_2 = [0.0, 0.2]   # Peak between 0.0V and 0.2V

    for start, end in cycles:
        E = potential[start:end]
        I = current[start:end]
        
        cycle_peak_data = {}

        # --- Find Anodic Peak ---
        anodic_mask = (E >= anodic_window[0]) & (E <= anodic_window[1])
        if np.any(anodic_mask):
            anodic_indices, _ = find_peaks(I[anodic_mask])
            if anodic_indices.size > 0:
                # Find the index of the max peak within the masked data
                max_peak_local_idx = anodic_indices[np.argmax(I[anodic_mask][anodic_indices])]
                # Convert back to an index in the full cycle data
                max_peak_global_idx = np.where(anodic_mask)[0][max_peak_local_idx]
                cycle_peak_data['Anodic_E(V)'] = E[max_peak_global_idx]
                cycle_peak_data['Anodic_I(A)'] = I[max_peak_global_idx]

        # --- Find 1st Cathodic Peak ---
        cathodic_mask_1 = (E >= cathodic_window_1[0]) & (E <= cathodic_window_1[1])
        if np.any(cathodic_mask_1):
            cathodic_indices_1, _ = find_peaks(-I[cathodic_mask_1])
            if cathodic_indices_1.size > 0:
                min_peak_local_idx = cathodic_indices_1[np.argmin(I[cathodic_mask_1][cathodic_indices_1])]
                min_peak_global_idx = np.where(cathodic_mask_1)[0][min_peak_local_idx]
                cycle_peak_data['Cathodic_1_E(V)'] = E[min_peak_global_idx]
                cycle_peak_data['Cathodic_1_I(A)'] = I[min_peak_global_idx]

        # --- Find 2nd Cathodic Peak ---
        cathodic_mask_2 = (E >= cathodic_window_2[0]) & (E <= cathodic_window_2[1])
        if np.any(cathodic_mask_2):
            cathodic_indices_2, _ = find_peaks(-I[cathodic_mask_2])
            if cathodic_indices_2.size > 0:
                min_peak_local_idx = cathodic_indices_2[np.argmin(I[cathodic_mask_2][cathodic_indices_2])]
                min_peak_global_idx = np.where(cathodic_mask_2)[0][min_peak_local_idx]
                cycle_peak_data['Cathodic_2_E(V)'] = E[min_peak_global_idx]
                cycle_peak_data['Cathodic_2_I(A)'] = I[min_peak_global_idx]

        all_cycle_peaks.append(cycle_peak_data)

    peak_data = pd.DataFrame({
        'Cycle': [f'Cycle {i+1}' for i in range(len(cycles))],
        'Anodic_E(V)': [p.get('Anodic_E(V)', np.nan) for p in all_cycle_peaks],
        'Anodic_I(A)': [p.get('Anodic_I(A)', np.nan) for p in all_cycle_peaks],
        'Cathodic_1_E(V)': [p.get('Cathodic_1_E(V)', np.nan) for p in all_cycle_peaks],
        'Cathodic_1_I(A)': [p.get('Cathodic_1_I(A)', np.nan) for p in all_cycle_peaks],
        'Cathodic_2_E(V)': [p.get('Cathodic_2_E(V)', np.nan) for p in all_cycle_peaks],
        'Cathodic_2_I(A)': [p.get('Cathodic_2_I(A)', np.nan) for p in all_cycle_peaks],
    })    
    # --- Save CSV ---
    cv_base = os.path.splitext(os.path.basename(cv_path))[0]
    output_dir = os.path.join('results', 'data')
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{cv_base}_peaks_SCE.csv")
    peak_data.to_csv(out_csv, index=False)
    return peak_data

def rhe(pH, cv_path):
    potential, current, cycles, _ = utils.read_data_cv(cv_path)
    # Calculate for the RHE conversion
    E_half = []
    for (start, end) in cycles:
        E = potential[start:end]; I = current[start:end]
        E_half.append((E[np.argmax(I)] + E[np.argmin(I)]) / 2)

    E_half_mean_SCE = np.mean(E_half)
    E_Fc_NHE = 0.63
    E_sce_to_nhe = E_Fc_NHE - E_half_mean_SCE
    E_sce_to_rhe = E_sce_to_nhe + 0.059 * pH

    # Save json
    json_filename = f"RHE_{os.path.splitext(os.path.basename(cv_path))[0]}.json"
    results = {
        "E_half_mean_SCE": E_half_mean_SCE,
        "E_sce_to_rhe": E_sce_to_rhe,
        "E_sce_to_nhe": E_sce_to_nhe,
        "pH": pH
    }
    utils.save_json(results, json_filename)