import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_pretreatment(cv_path, tag_date):
    """
    Plot the pre-treatment cyclic voltammogram from the given CV file.
    Each scan is separated and plotted on the same figure.

    Parameters
    ----------
    cv_path : str
        Path to the pre-treatment CV text file.
    tag_date : str
        Tag used for naming the output figure (e.g. experiment date).
    """
    # --- Read the data ---
    data = pd.read_csv(cv_path, sep=';', decimal=',')
    potential = data['WE(1).Potential (V)'].values
    current = data['WE(1).Current (A)'].values
    scans = data['Scan'].values
    uniq_scans = pd.unique(scans)

    # --- Find start and end indices for each scan ---
    cycles = [(np.where(scans == s)[0][0], np.where(scans == s)[0][-1] + 1) for s in uniq_scans]

    # --- Plot each scan ---
    plt.figure(figsize=(8, 6))
    for (start, end), s in zip(cycles, uniq_scans):
        plt.plot(potential[start:end], current[start:end], label=f'Scan {s}')

    plt.xlabel('Potential (V vs SCE)')
    plt.ylabel('Current (A)')
    plt.title('Pre-treatment CV Scans')
    plt.legend()
    plt.tight_layout()

    # --- Save the figure ---
    results_fig_dir = "results/figures"
    os.makedirs(results_fig_dir, exist_ok=True)
    fig_path = os.path.join(results_fig_dir, f"pretreatment_{tag_date}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"✔ Pretreatment CV plot saved to {fig_path}")