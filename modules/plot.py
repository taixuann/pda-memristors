import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from pathlib import Path
from modules import utils
from modules.support import srd_support, raw_cv_support
from adjustText import adjust_text


def scan_rate_dependency(scan_rates_mv, data_folder, filename_template, file_save, **kwargs):

    # Construct the paths using the helper function
    cv_paths = srd_support.construct_multiple_paths(scan_rates_mv, filename_template, data_dir=utils.RAW_DATA / data_folder, **kwargs)
    scan_rates_v = [rate / 1000.0 for rate in scan_rates_mv]
    sqrt_scan_rates, anodic_peaks_uA, cathodic_peaks_1_uA, cathodic_peaks_2_uA = srd_support.randles_sevnick(scan_rates_v, cv_paths)

    def verify_peaks(verify=True):
        srd_support.plot_all_cvs_with_peaks(cv_paths, scan_rates_mv, file_save, cycle_number=0)
        for path in cv_paths:
            srd_support.first_cv_with_peaks(path, cycle_number=0)
        return verify
    
    def plot_ipeak_scan_rate():
        # Convert scan rates from mV/s to V/s for calculation
    
        anodic_fit, anodic_r2, cathodic_1_fit, cathodic_1_r2, cathodic_2_fit, cathodic_2_r2 = srd_support.regression_line(sqrt_scan_rates, anodic_peaks_uA, cathodic_peaks_1_uA, cathodic_peaks_2_uA)

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(sqrt_scan_rates, anodic_peaks_uA, 'o-', label='Anodic Peak Current')
        plt.plot(sqrt_scan_rates, anodic_fit, '--', color='tab:blue', label=f'Anodic Fit ($R^2={anodic_r2:.4f}$)')
        plt.plot(sqrt_scan_rates, cathodic_peaks_1_uA, 's-', label='Cathodic Peak 1')
        plt.plot(sqrt_scan_rates, cathodic_1_fit, '--', color='tab:orange', label=f'Cathodic Fit 1 ($R^2={cathodic_1_r2:.4f}$)')
        plt.plot(sqrt_scan_rates, cathodic_peaks_2_uA, '^-', label='Cathodic Peak 2')
        plt.plot(sqrt_scan_rates, cathodic_2_fit, '--', color='tab:green', label=f'Cathodic Fit 2 ($R^2={cathodic_2_r2:.4f}$)')

        # --- Annotate points with current values ---
        texts = []
        # Anodic annotations
        for x, y in zip(sqrt_scan_rates, anodic_peaks_uA):
            texts.append(plt.text(x, y, f'{y:.2f}', color='tab:blue', fontsize=9))
        # Cathodic annotations
        for x, y in zip(sqrt_scan_rates, cathodic_peaks_1_uA):
            texts.append(plt.text(x, y, f'{y:.2f}', color='tab:orange', fontsize=9))
        for x, y in zip(sqrt_scan_rates, cathodic_peaks_2_uA):
            texts.append(plt.text(x, y, f'{y:.2f}', color='tab:green', fontsize=9))
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        plt.title("Au, Pt, SCE (pH = 8.4)")
        plt.xlabel(r"$\nu^{1/2} \ (V/s)^{1/2}$")
        plt.ylabel(r"$I_{peak} \ (\mu A)$")
        plt.grid(True, linestyle='--', alpha=0.5)
        # Adjust legend to prevent overlap
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        output_dir = utils.SUMMARY_DIR / "randles_sevcik_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_file = output_dir / f"{file_save}_RS_plot.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_scan_rate_vs_current_anodic_peak_comparison(scan_rates_mv, data_folder, cv_paths):
        """
        Generates a single Randles-Sevcik plot comparing anodic peak currents
        across different experimental conditions (pH and electrodes).
        """
        conditions = [
            {'ph': 7.4, 'electrodes': 'Au, Pt, SCE'},
            {'ph': 8.4, 'electrodes': 'Au, Pt, SCE'},
            {'ph': 7.4, 'electrodes': 'GC, Pt, SCE'},
            {'ph': 8.4, 'electrodes': 'GC, Pt, SCE'},
        ]
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
        texts = []

        for i, cond in enumerate(conditions):
            ph_value = cond['ph']
            electrodes = cond['electrodes']
            label = f"pH {ph_value}, {electrodes.split(',')[0]}"
            color = colors[i]
            filename_template = "PDA_{electrodes}_{scan_rate}mV_pH {ph_value}.txt"
            cv_paths = srd_support.construct_multiple_paths(scan_rates_mv, filename_template, data_dir=utils.RAW_DATA / data_folder, electrodes=electrodes, ph_value=ph_value)
            
            sqrt_scan_rates, anodic_peaks_uA, _, _ = srd_support.randles_sevnick(scan_rates_v, cv_paths)

            # --- Linear fit and R^2 ---
            slope, intercept = np.polyfit(sqrt_scan_rates, anodic_peaks_uA, 1)
            r2 = np.corrcoef(sqrt_scan_rates, anodic_peaks_uA)[0, 1]**2
            fit_line = slope * np.array(sqrt_scan_rates) + intercept

            # --- Plotting ---
            plt.plot(sqrt_scan_rates, anodic_peaks_uA, 'o', color=color, label=label)
            plt.plot(sqrt_scan_rates, fit_line, '--', color=color, label=f'Fit ($R^2={r2:.4f}$)')
            # --- Annotate points with current values ---
            for x, y in zip(sqrt_scan_rates, anodic_peaks_uA):
                texts.append(plt.text(x, y, f'{y:.2f}', color=color, fontsize=9))
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7))

        plt.title("Anodic Peak Current vs. Scan Rate Comparison")
        plt.xlabel(r"$\nu^{1/2} \ (V/s)^{1/2}$")
        plt.ylabel(r"$I_{peak, anodic} \ (\mu A)$")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Experimental Condition", loc='best')
        plt.tight_layout()
        output_dir = utils.SUMMARY_DIR / "anodic_peak_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_file = output_dir / "anodic_peak_comparison_RS_plot.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✔ Anodic peak comparison plot saved to {fig_file.name}")
    
    def plot_cv_scan_rate_pH(scan_rates_mv, data_folder):
        """
        Plots the first cycle of CV scans for different pH values and electrodes
        at a specific scan rate in a single figure.
        """
        scan_rate = 10
        electrode_types = ['Au', 'GC']
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        fig.suptitle(f"First Two CV Cycles Comparison at {scan_rate} mV/s", fontsize=16)

        for i, electrode_type in enumerate(electrode_types):
            ax = axes[i]
            texts = [] # Store text objects for adjust_text
            conditions = [
                {'ph': 7.4, 'electrodes': f'{electrode_type}, Pt, SCE', 'scan_rate': scan_rate},
                {'ph': 8.4, 'electrodes': f'{electrode_type}, Pt, SCE', 'scan_rate': scan_rate},
            ]
            # Use the centralized color selection function to get blue and red
            colors = [utils.color_selection_plot('blue'), utils.color_selection_plot('red')]
            
            for j, cond in enumerate(conditions):
                ph_value = cond['ph']
                electrodes = cond['electrodes']
                scan_rate = cond['scan_rate']
                label = f"pH {ph_value}"
                color = colors[j]

                filename_template = "PDA_{electrodes}_{scan_rate}mV_pH {ph_value}.txt"
                cv_path = srd_support.construct_single_path(scan_rate, filename_template, data_dir=utils.RAW_DATA / data_folder, electrodes=electrodes, ph_value=ph_value)
                
                potential, current, cycles, _ = utils.read_data_cv(cv_path)
                first_cycle_start, first_cycle_end = cycles[0]
                second_cycle_start, second_cycle_end = cycles[1]
                
                # Plot the first cycle (solid line) and second cycle (dashed line)
                ax.plot(potential[first_cycle_start:first_cycle_end], current[first_cycle_start:first_cycle_end] * 1e6, color=color, label=f"{label} (cycle 1)", linestyle='-')
                ax.plot(potential[second_cycle_start:second_cycle_end], current[second_cycle_start:second_cycle_end] * 1e6, color=color, label=f"{label} (cycle 2)", linestyle=':')

                # --- Find and plot peaks for the first cycle ---
                peak_data = raw_cv_support.save_peaks_cv(cv_path)
                first_cycle_peaks = peak_data.iloc[0]

                peak_info = {
                    'Anodic': ('Anodic_E(V)', 'Anodic_I(A)'),
                    'Cathodic 1': ('Cathodic_1_E(V)', 'Cathodic_1_I(A)'),
                    'Cathodic 2': ('Cathodic_2_E(V)', 'Cathodic_2_I(A)')
                }

                for peak_name, (e_col, i_col) in peak_info.items():
                    E_peak = first_cycle_peaks.get(e_col)
                    I_peak = first_cycle_peaks.get(i_col)
                    if pd.notna(E_peak) and pd.notna(I_peak):
                        ax.axvline(x=E_peak, ymin=0, ymax=0.95, color=color, linestyle='--', lw=1.2)
                        # Append text object to list for later adjustment
                        texts.append(ax.text(E_peak, -0.05, f'{E_peak:.2f}V', 
                                             color=color, ha='center', va='top', fontsize=9,
                                             transform=ax.get_xaxis_transform()))
            
            # Automatically adjust text to avoid overlap
            # This will move labels horizontally to prevent them from colliding
            adjust_text(texts, ax=ax, only_move={'text': 'x'},
                        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
            
            # Adjust bottom margin to make space for the text
            plt.subplots_adjust(bottom=0.15)
            ax.set_title(f"{electrode_type} electrode")
            ax.set_xlabel("Potential (V vs SCE)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title="Condition", loc='upper left', fontsize='small')

        axes[0].set_ylabel("Current (μA)")

        plt.tight_layout()
        output_dir = utils.SUMMARY_DIR / "cv_comparison_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_file = output_dir / f"{scan_rate}_first_two_cycles_cv_comparison.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✔ CV comparison plot saved to {fig_file.name}")
    #plot_cv_scan_rate_pH(scan_rates_mv)

    def table_current_potential_comparison(scan_rates_mv):
        """
        Extracts peak current and potential for different experimental conditions
        and saves them into a summary CSV file.
        """
        conditions = [
            {'ph': 7.4, 'electrodes': 'Au, Pt, SCE'},
            {'ph': 8.4, 'electrodes': 'Au, Pt, SCE'},
            {'ph': 7.4, 'electrodes': 'GC, Pt, SCE'},
            {'ph': 8.4, 'electrodes': 'GC, Pt, SCE'},
        ]
        all_conditions_data = []

        for cond in conditions:
            ph_value = cond['ph']
            electrodes = cond['electrodes']
            filename_template = "PDA_{electrodes}_{scan_rate}mV_pH {ph_value}.txt"
            data_folder = "scan_rate_dependency"
            cv_paths = srd_support.construct_multiple_paths(scan_rates_mv, filename_template, data_dir=utils.RAW_DATA / data_folder, electrodes=electrodes, ph_value=ph_value)

            for i, cv_path in enumerate(cv_paths):
                if not cv_path.exists():
                    print(f"[Warning] File not found, skipping: {cv_path.name}")
                    continue
                
                peak_data = raw_cv_support.save_peaks_cv(cv_path)
                first_cycle_peaks = peak_data.iloc[0].to_dict()

                # --- Convert current to µA ---
                if pd.notna(first_cycle_peaks.get('Anodic_I(A)')):
                    first_cycle_peaks['Anodic_I(µA)'] = first_cycle_peaks['Anodic_I(A)'] * 1e6
                if pd.notna(first_cycle_peaks.get('Cathodic_1_I(A)')):
                    first_cycle_peaks['Cathodic_1_I(µA)'] = first_cycle_peaks['Cathodic_1_I(A)'] * 1e6
                # Calculate the ratio of Ipc/Ipa for the first cathodic peak
                ipa = first_cycle_peaks.get('Anodic_I(A)', np.nan)
                ipc = first_cycle_peaks.get('Cathodic_1_I(A)', np.nan)
                if pd.notna(ipa) and pd.notna(ipc) and ipa != 0:
                    first_cycle_peaks['Ratio Ipc/Ipa'] = abs(ipc / ipa)
                else:
                    first_cycle_peaks['Ratio Ipc/Ipa'] = np.nan

                first_cycle_peaks['pH'] = ph_value
                first_cycle_peaks['Electrode'] = electrodes.split(',')[0]
                first_cycle_peaks['Scan Rate (mV/s)'] = scan_rates_mv[i]
                all_conditions_data.append(first_cycle_peaks)

        summary_df = pd.DataFrame(all_conditions_data)
        # Drop the now-unnecessary 'Cycle' column
        if 'Cycle' in summary_df.columns:
            summary_df = summary_df.drop(columns=['Cycle'])

        # --- Create a single pivot table for all electrodes ---
        pivot_df = summary_df.pivot_table(
            index=['Electrode', 'Scan Rate (mV/s)'],
            columns='pH',
            values=[
                'Anodic_E(V)', 'Anodic_I(µA)',
                'Cathodic_1_E(V)', 'Cathodic_1_I(µA)',
                'Ratio Ipc/Ipa'
            ]
        )

        # --- Reorder and Rename Columns for Clarity ---
        if not pivot_df.empty:
            # Reorder levels to have pH on top
            pivot_df = pivot_df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
            
            # Define shorter, more readable names for the value columns
            new_level_names = {
                'Anodic_E(V)': 'Epa\n(V)',
                'Anodic_I(µA)': 'Ipa\n(µA)',
                'Cathodic_1_E(V)': 'Epc\n(V)',
                'Cathodic_1_I(µA)': 'Ipc\n(µA)',
                'Ratio Ipc/Ipa': 'Ipc/Ipa\nRatio'
            }
            pivot_df = pivot_df.rename(columns=new_level_names, level=1)

        # --- Save the merged table to a single CSV file ---
        output_dir = utils.SUMMARY_DIR / "peak_summary_tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / "peak_summary_table_merged.csv"
        pivot_df.to_csv(output_csv)
        print(f"✔ Merged peak summary table saved to {output_csv.name}")

        # --- Visualize the table as a professional-looking image ---
        if not pivot_df.empty:
            # Prepare data for plotting
            plot_df = pivot_df.map(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
            
            fig, ax = plt.subplots(figsize=(10, 5)) # Adjust size for a wider table
            ax.axis('off')

            # --- Create the table ---
            col_labels = [f"{sub_col}" for ph, sub_col in plot_df.columns] # Sub-column headers
            row_labels = plot_df.index.get_level_values('Scan Rate (mV/s)')
            
            table = ax.table(
                cellText=plot_df.values,
                colLabels=col_labels,
                rowLabels=row_labels,
                cellLoc='center',
                loc='center',
                colWidths=[0.08] * len(col_labels)
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.5, 2)

            # --- Style the table and add merged headers ---
            n_scan_rates = len(scan_rates_mv)
            n_cols_per_ph = 5

            for (i, j), cell in table.get_celld().items():
                # Header rows
                if i == 0:
                    cell.set_text_props(weight='bold')
                # Zebra striping for data rows (i > 0)
                elif i % 2 == 0:
                    cell.set_facecolor('#f2f2f2')

            # Add merged headers for pH
            header_y_pos = 1.0 / (len(plot_df) + 2) * (len(plot_df) + 1.5)
            ax.text(0.375, header_y_pos, 'pH 7.4', transform=ax.transAxes, ha='center', va='bottom', weight='bold', fontsize=12)
            ax.text(0.775, header_y_pos, 'pH 8.4', transform=ax.transAxes, ha='center', va='bottom', weight='bold', fontsize=12)

            # Add merged row labels for Electrodes
            electrodes = plot_df.index.get_level_values('Electrode').unique()
            for i, electrode in enumerate(electrodes):
                row_start = i * n_scan_rates
                # Calculate y position to center the text over the merged rows
                y_pos = 1.0 / (len(plot_df) + 1) * (len(plot_df) - row_start - n_scan_rates / 2.0 + 0.5)
                ax.text(-0.25, y_pos, electrode, transform=ax.transAxes, ha='center', va='center', weight='bold', fontsize=12)

            # Add headers for the first two columns
            ax.text(-0.25, header_y_pos, 'Electrode', transform=ax.transAxes, ha='center', va='bottom', weight='bold', fontsize=10)
            ax.text(-0.05, header_y_pos, 'Scan Rate\n(mV/s)', transform=ax.transAxes, ha='center', va='bottom', weight='bold', fontsize=10)

            # Hide original row labels as we've replaced them
            cells = table.get_celld()
            for i in range(len(plot_df) + 1): # +1 to include header row
                if (i, -1) in cells: # Check if the row label cell exists
                    cells[(i, -1)].set_text_props(visible=False)

            plt.title("Peak Summary Comparison", fontsize=16, weight='bold', y=0.9)
            output_png = output_dir / "peak_summary_table_merged.png"
            plt.savefig(output_png, dpi=300, bbox_inches='tight')
            print(f"✔ Merged peak summary table image saved to {output_png.name}")
    
    def calculate_net_deposition(data_folder, electrode_area_cm2, n_electrons=2):
        """
        Calculates and prints the net molecular deposition for various experimental conditions.
        This function iterates through predefined conditions (pH, electrodes) and calculates
        the net charge and molecule accumulation on a cycle-by-cycle basis for a
        representative CV file (at 100 mV/s).
        """
        print("\n--- Net Deposition Analysis ---")
        FARADAY_CONSTANT = 96485  # C/mol
        AVOGADRO_CONSTANT = 6.022e23  # molecules/mol
        PDA_MOLAR_MASS = 147.13  # g/mol (assuming C8H5NO2 monomer unit)
        PDA_DENSITY = 1.5  # g/cm^3

        scan_rate_mv = 10  # Use a single representative scan rate
        scan_rate_v = scan_rate_mv / 1000.0

        # Data storage for plotting
        deposition_rate_data = {}
        total_growth_data = {}
        thickness_data = {}

        conditions = [
            {'ph': 7.4, 'electrodes': 'Au, Pt, SCE'},
            {'ph': 8.4, 'electrodes': 'Au, Pt, SCE'},
            {'ph': 7.4, 'electrodes': 'GC, Pt, SCE'},
            {'ph': 8.4, 'electrodes': 'GC, Pt, SCE'},
        ]

        for cond in conditions:
            label = f"pH {cond['ph']}, {cond['electrodes'].split(',')[0]}"
            filename_template = "PDA_{electrodes}_{scan_rate}mV_pH {ph_value}.txt"
            cv_path = srd_support.construct_single_path(
                scan_rate_mv, filename_template,
                data_dir=utils.RAW_DATA / data_folder,
                electrodes=cond['electrodes'], ph_value=cond['ph']
            )

            if not cv_path.exists():
                print(f"\n[Warning] File not found, skipping: {cv_path.name}")
                continue

            potential, current, cycles, _ = utils.read_data_cv(cv_path)
            print(f"\n=== Analysis for {label} ===")
            print(f"File: {cv_path.name}")

            total_accumulated_molecules = 0
            cycle_numbers = []
            molecules_per_cycle = []
            growth_per_cycle = []
            thickness_per_cycle = []
            for i, (start_idx, end_idx) in enumerate(cycles):
                cycle_numbers.append(i + 1)
                E_cycle = potential[start_idx:end_idx]
                I_cycle = current[start_idx:end_idx]

                # Find the switching potential to split forward and backward scans
                vertex_idx = np.argmax(E_cycle)
                E_fwd, I_fwd = E_cycle[:vertex_idx], I_cycle[:vertex_idx]
                E_bwd, I_bwd = E_cycle[vertex_idx:], I_cycle[vertex_idx:]

                # Integrate to get charge, ensuring we handle empty arrays
                Q_anodic = abs(np.trapz(I_fwd, E_fwd)) / scan_rate_v if E_fwd.size > 1 else 0
                Q_cathodic = abs(np.trapz(I_bwd, E_bwd)) / scan_rate_v if E_bwd.size > 1 else 0

                Q_net = Q_anodic - Q_cathodic
                if Q_net < 0:
                    Q_net = 0

                moles_newly_deposited = Q_net / (n_electrons * FARADAY_CONSTANT)
                molecules_newly_deposited = moles_newly_deposited * AVOGADRO_CONSTANT
                total_accumulated_molecules += molecules_newly_deposited
                current_coverage = total_accumulated_molecules / electrode_area_cm2
                
                # --- Thickness Calculation ---
                # Thickness (cm) = (Total Molecules / Area) * (1 / Avogadro) * Molar Mass / Density
                moles_per_cm2 = current_coverage / AVOGADRO_CONSTANT
                volume_per_cm2 = (moles_per_cm2 * PDA_MOLAR_MASS) / PDA_DENSITY # This is thickness in cm
                thickness_nm = volume_per_cm2 * 1e7 # Convert cm to nm

                molecules_per_cycle.append(molecules_newly_deposited)
                growth_per_cycle.append(total_accumulated_molecules)
                thickness_per_cycle.append(thickness_nm)

                print(f"Cycle {i+1}:")
                print(f"   Q_anodic (Built): {Q_anodic:.2e} C")
                print(f"   Q_cathodic (Lost): {Q_cathodic:.2e} C")
                print(f"   Q_net (Stuck):     {Q_net:.2e} C")
                print(f"   -> Molecules Added: {molecules_newly_deposited:.2e}")
                print(f"   -> Total Coverage:  {current_coverage:.2e} molecules/cm²")
                print(f"   -> Film Thickness:  {thickness_nm:.3f} nm")
            
            deposition_rate_data[label] = (cycle_numbers, molecules_per_cycle)
            total_growth_data[label] = (cycle_numbers, growth_per_cycle)
            thickness_data[label] = (cycle_numbers, thickness_per_cycle)

        # --- Plotting Deposition Rate ---
        plt.figure(figsize=(10, 7))
        for label, (cycles, rates) in deposition_rate_data.items():
            plt.plot(cycles, rates, marker='o', linestyle='-', label=label)
        plt.xlabel("Cycle Number")
        plt.ylabel("Molecules Deposited per Cycle")
        plt.title(f"Deposition Rate per Cycle at {scan_rate_mv} mV/s")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        output_dir = utils.SUMMARY_DIR / "deposition_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        rate_fig_path = output_dir / f"deposition_rate_per_cycle_{scan_rate_mv}mV.png"
        plt.savefig(rate_fig_path, dpi=300)
        plt.show()
        print(f"✔ Deposition rate plot saved to {rate_fig_path.name}")

        # --- Plotting Total Film Growth ---
        plt.figure(figsize=(10, 7))
        for label, (cycles, growth) in total_growth_data.items():
            plt.plot(cycles, growth, marker='s', linestyle='-', label=label)
        plt.xlabel("Cycle Number")
        plt.ylabel("Total Accumulated Molecules")
        plt.title(f"Total Film Growth vs. Cycles at {scan_rate_mv} mV/s")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        output_dir = utils.SUMMARY_DIR / "deposition_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        growth_fig_path = output_dir / f"total_film_growth_{scan_rate_mv}mV.png"
        plt.savefig(growth_fig_path, dpi=300)
        plt.show()
        print(f"✔ Total film growth plot saved to {growth_fig_path.name}")

        # --- Plotting Film Thickness ---
        plt.figure(figsize=(10, 7))
        for label, (cycles, thickness) in thickness_data.items():
            plt.plot(cycles, thickness, marker='D', linestyle='-', label=label)
        plt.xlabel("Cycle Number")
        plt.ylabel("Estimated Film Thickness (nm)")
        plt.title(f"Estimated Film Thickness vs. Cycles at {scan_rate_mv} mV/s")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        output_dir = utils.SUMMARY_DIR / "deposition_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        thickness_fig_path = output_dir / f"film_thickness_vs_cycles_{scan_rate_mv}mV.png"
        plt.savefig(thickness_fig_path, dpi=300)
        plt.show()
        print(f"✔ Film thickness plot saved to {thickness_fig_path.name}")
        
    return table_current_potential_comparison(scan_rates_mv)


def raw_cv_plot(cv_path, pH):
    potential, current, cycles, num_cycles = utils.read_data_cv(cv_path)

    def cv_analyze(highlight_scans=False):
        fig_file = utils.FIGURES_DIR / f"{Path(cv_path).stem}_plot_SCE.png"
        raw_cv_support._create_plot(potential, current, cycles, num_cycles,
                     title="",
                     xlabel="Potential (V vs SCE)",
                     ylabel="Current (A)",
                     fig_path=fig_file)
        raw_cv_support.save_peaks_cv(cv_path)
        if highlight_scans:
            raw_cv_support.plot_highlight(cv_path)

    cv_analyze(highlight_scans=False)
    #calibration_analyze(pH)
    #pre_treat()
