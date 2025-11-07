#!/usr/bin/env python3
"""
main.py
-------
Top-level entry point for the PDA Memristor CV analysis pipeline.

This script runs the full workflow:
1. Calibration (Fc/Fc+ vs SCE → RHE)
2. Pre-treatment CV plotting
3. PDA electropolymerization plotting
4. Electrochemical calculations (ΔEp, i_ox/i_red, n)

Usage examples:
---------------
# Run everything:
python main.py --all --date "03-11-25" --pH 7.4 --scan --electrode --condition --exp

# Run only plotting:
python main.py --plot_electropoly --pda_path "PDA_1_Au, Pt small, SCE_10mV_2-10-25.txt"

# Run only calculations:
python main.py --calc --date "03-11-25"
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from modules.plotting import calibration, electropoly, pretreat
from modules.analysis import peaks
from modules import utils


def main():
    parser = argparse.ArgumentParser(description = "PDA Memristor CV Analysis Pipeline")

    # Experiment metadata
    parser.add_argument("--sce_path", type = str, help = "Filename of calibration data")
    parser.add_argument("--date", type = str, help = "Experiment date tag (e.g., 03-11-25)")
    parser.add_argument("--pH", type = float, help = "Solution pH for RHE conversion")
    parser.add_argument("--pda_path", type = str, help = "Filename of electropolymerization for PDA")
    parser.add_argument("--pretreat_filename", type=str, help="Filename of pre-treatment data")

    #Option arguement
    parser.add_argument("--pdapeaks", type = str, help = "Turn on of turn off the function of runnign pdapeaks")
    # Actions
    parser.add_argument("--plot_electropoly", action = "store_true", help = "Run plotting electropolymerization only")
    parser.add_argument("--plot_pretreat", action = "store_true", help = "Run plotting pre-treat only")
    parser.add_argument("--plot_calibration", action = "store_true", help = "Run plotting calibration only")
    parser.add_argument("--calc", action = "store_true", help = "Run electrochemical calculations only")   
    parser.add_argument("--all", action = "store_true", help = "Run full pipeline")
    parser.add_argument("--show-config", action = "store_true", help = "Print project configuration")

    args = parser.parse_args()

    # --- Argument validation based on action ---
    action_selected = any([args.all, args.plot_electropoly, args.plot_pretreat, args.plot_calibration, args.calc])

    # Run pipeline
    if args.all:
        if not all([args.sce_path, args.date, args.pH, args.pda_path, args.pretreat_filename]):
            parser.error("--all requires --sce_path, --date, --pH, --pda_path, and --pretreat_filename.")
        print("=== Running full workflow ===")
        calibration.run_calibration(args.sce_path, args.pH, args.date)
        if args.pretreat_filename:
            pretreat.plot_pretreatment(args.pretreat_filename, args.date)
        electropoly.run_electropolymerization_SCE(args.pda_path)
        peaks.extract_and_save_SCE(args.pda_path)
    
    elif args.plot_electropoly:
        if not args.pda_path:
            parser.error("--plot_electropoly requires --pda_path.")
        print("=== Running plotting electropolymerization only ===")
        electropoly.run_electropolymerization_SCE(args.pda_path)
        print("\n=== Running peak extraction for electropolymerization ===")
        peaks.extract_and_save_SCE(args.pda_path)
    
    elif args.plot_pretreat:
        if not (args.pretreat_filename and args.date):
            parser.error("--plot_pretreat requires --pretreat_filename and --date.")
        print("=== Running plotting pre-treatment only ===")
        pretreat.plot_pretreatment(args.pretreat_filename, args.date)

    elif args.plot_calibration:
        if not all([args.sce_path, args.pH]):
            parser.error("--plot_calibration requires --sce_path, --pH, and --date.")
        print("=== Running plotting calibration and calculation of calibration ===")
        calibration.run_calibration(args.sce_path, args.pH)
    
    elif not action_selected:
        print("⚠ No action selected. Use --plot_electropoly, --plot_pretreat, --plot_calibration, or --all.")
        parser.print_help()

if __name__ == "__main__":
    main()
   
