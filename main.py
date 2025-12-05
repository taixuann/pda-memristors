#!/usr/bin/env python3
import argparse
from modules import plot
from modules import utils


def main():
    parser = argparse.ArgumentParser(description = "PDA Memristor CV Analysis Pipeline")

    # Experiment metadata
    parser.add_argument("--cv_path", type = str, help = "Filename of cv data")
    parser.add_argument("--pH", type = float, help = "Solution pH for RHE conversion")

    args = parser.parse_args()

    # --- Parameter ---
    # Define the parameters for your experiments

    # Single file raw plotting
    cv_path = "/Users/tai/Downloads/Research/PDA-based memristors/data_analysis_python/raw_data/5-12-25/PDA_GC_pH 7.4 (PO4-)_5-12-25.txt"
    pH = 7.4

    # --- Running (comment out for process won't run) ---
    # plot.cv_analyze(cv_path, highlight_scans=False)
    
    # --- Scan rate dependency for plot_ipeak_scan_rate only --- 
    electrodes = "Au, Pt, SCE"
    ph_value = 8.4
    scan_rates_mv = [10, 15, 20, 50, 100] # Assuming same scan rates for all experiments
    data_folder = "scan_rate_dependency"
    filename_template = "PDA_{electrodes}_{scan_rate}mV_pH {ph_value}.txt"
    file_save = f"PDA_{electrodes}_pH{ph_value}"
    
    # --- Running ---
    #plot.scan_rate_dependency(scan_rates_mv, data_folder, filename_template, file_save, electrodes=electrodes, ph_value=ph_value)
    plot.raw_cv_plot(cv_path,pH)

if __name__ == "__main__":
    main()
   
