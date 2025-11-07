import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from modules.utils import RAW_DATA


def extract_and_save_SCE(pda_filename):
    """
    Extract anodic and cathodic peaks from CV file, save to CSV and print summary.
    Args:
        cv_path (str): Path to CV .csv file.
        E_sce_to_rhe (float): Offset to convert SCE to RHE.
        tag_date (str): Date tag for output CSV file naming.
    """
    # --- Read CV data ---
    pda_path = RAW_DATA / pda_filename
    data_cv = pd.read_csv(pda_path, sep=';', decimal=',')
    if 'WE(1).Potential (V)' not in data_cv.columns or 'WE(1).Current (A)' not in data_cv.columns:
        raise ValueError("Expected columns 'WE(1).Potential (V)' and 'WE(1).Current (A)' in CV file.")
    potential = data_cv['WE(1).Potential (V)'].values
    current = data_cv['WE(1).Current (A)'].values
    # --- Detect cycles ---
    cycles = []
    scan_col = None
    for col in ['Scan', 'scan']:
        if col in data_cv.columns:
            scan_col = col
            break
    if scan_col:
        scans = data_cv[scan_col].values
        uniq_scans = pd.unique(scans)
        for s in uniq_scans:
            idxs = np.where(scans == s)[0]
            if idxs.size > 0:
                cycles.append((int(idxs[0]), int(idxs[-1]) + 1))
    else:
        # Fallback: detect by potential reversals
        dV = np.diff(potential)
        direction = np.sign(dV)
        reversals = np.where(np.diff(direction) != 0)[0]
        reversals = [0] + reversals.tolist() + [len(potential)]
        cycles = [(reversals[i], reversals[i+1]) for i in range(len(reversals)-1)]
    # --- Find peaks for each cycle ---
    anodic_peaks = []
    cathodic_peaks = []
    anodic_peaks_rhe = []
    cathodic_peaks_rhe = []
    for start, end in cycles:
        E = potential[start:end]
        I = current[start:end]
        # Find peaks
        if len(I) > 0:
            peaks_anodic, _ = find_peaks(I)
            peaks_cathodic, _ = find_peaks(-I)
            if len(peaks_anodic) > 0:
                max_peak_idx = peaks_anodic[np.argmax(I[peaks_anodic])]
                anodic_peaks.append((float(E[max_peak_idx]), float(I[max_peak_idx])))
            else:
                anodic_peaks.append((np.nan, np.nan))
                anodic_peaks_rhe.append((np.nan, np.nan))
            if len(peaks_cathodic) > 0:
                min_peak_idx = peaks_cathodic[np.argmin(I[peaks_cathodic])]
                cathodic_peaks.append((float(E[min_peak_idx]), float(I[min_peak_idx])))
            else:
                cathodic_peaks.append((np.nan, np.nan))
                cathodic_peaks_rhe.append((np.nan, np.nan))
        else:
            anodic_peaks.append((np.nan, np.nan))
            cathodic_peaks.append((np.nan, np.nan))
            anodic_peaks_rhe.append((np.nan, np.nan))
            cathodic_peaks_rhe.append((np.nan, np.nan))
    # --- Prepare DataFrame ---
    peak_data = pd.DataFrame({
        'Cycle': [f'Cycle {i+1}' for i in range(len(cycles))],
        'Anodic_E_SCE(V)': [a[0] for a in anodic_peaks],
        'Anodic_I_SCE(A)': [a[1] for a in anodic_peaks],
        'Cathodic_E_SCE(V)': [c[0] for c in cathodic_peaks],
        'Cathodic_I_SCE(A)': [c[1] for c in cathodic_peaks],
    })
    # --- Save CSV ---
    pda_base = os.path.splitext(os.path.basename(pda_path))[0]
    output_dir = os.path.join('results', 'data')
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{pda_base}_peaks_SCE.csv")
    peak_data.to_csv(out_csv, index=False)
    # --- Print summary ---
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
    print(f"Peak data saved to {out_csv}")