#!/bin/bash
set -e

usage() {
    echo "Usage: $0 -a <action: all|calib|pretreat|pda|calc> -e <electrode> -n <experiment> -s <scan rate> -c <condition> -d <date> -p <pH>"
    echo "Example:"
    echo "  $0 -a all -e \"Au, Pt small, SCE\" -n 1 -s 10mV -c normal -d 02-10-25 -p 7.4"
    exit 1
}

# --- Parse arguments ---
while getopts "a:e:n:s:c:d:p:" opt; do
    case $opt in
        a) action="$OPTARG" ;;
        e) electrode="$OPTARG" ;;
        n) no_exp="$OPTARG" ;;
        s) scan_rate="$OPTARG" ;;
        c) condition="$OPTARG" ;;
        d) file_date="$OPTARG" ;;
        p) pH="$OPTARG" ;;
        *) usage ;;
    esac
done

# --- Check required arguments ---
if [ -z "$action" ] || [ -z "$electrode" ] || [ -z "$no_exp" ] || [ -z "$scan_rate" ] || [ -z "$condition" ] || [ -z "$file_date" ] || [ -z "$pH" ]; then
    usage
fi

# --- Base directories ---
base_dir="/Users/tai/Downloads/Research/PDA-based memristors/plotting"
raw_data="${base_dir}/raw_data"
results_dir="${base_dir}/results"
data_dir="${results_dir}/data"
figures_dir="${results_dir}/figures"
summary_dir="${results_dir}/summary"

# --- Sanitize electrode name for file paths ---
sanitize() {
    local s="$1"
    s="${s//,/_}"      # replace commas
    s="${s// /_}"      # replace spaces
    s="${s//\{/_}"     # replace {
    s="${s//\}/_}"     # replace }
    echo "$s"
}

safe_electrode=$(sanitize "$electrode")

# --- Common input paths (raw data) ---
sce_calib_path="${raw_data}/Fc_SCE_${safe_electrode}_${file_date}.txt"
pretreat_path="${raw_data}/Pre_treat_${no_exp}_${safe_electrode}_${condition}_${file_date}.txt"
cv_path="${raw_data}/PDA_${no_exp}_${safe_electrode}_${scan_rate}_${file_date}.txt"
calib_json_path="${data_dir}/RHE_Fc_SCE_${safe_electrode}_${file_date}.json"

# --- Ensure result folders exist ---
mkdir -p "$data_dir" "$figures_dir" "$summary_dir"

# --- Execute selected actions ---
if [[ "$action" == "all" || "$action" == "calib" ]]; then
    echo "=== [1] Running Calibration Plot for ${file_date} ==="
    python3 "${base_dir}/cv_plot.py" \
        --sce "${sce_calib_path}" \
        --pretreat "${pretreat_path}" \
        --cv "${cv_path}" \
        --pH "${pH}" \
        --tags-date "${file_date}" \
        --out-data "${data_dir}" \
        --out-fig "${figures_dir}"
fi

if [[ "$action" == "all" || "$action" == "pretreat" ]]; then
    echo "=== [2] Plotting Pre-treatment CV ==="
    python3 "${base_dir}/cv_plot.py" \
        --pretreat "${pretreat_path}" \
        --pH "${pH}" \
        --tags-date "${file_date}" \
        --out-fig "${figures_dir}"
fi

if [[ "$action" == "all" || "$action" == "pda" ]]; then
    echo "=== [3] Plotting PDA Electropolymerization CV ==="
    python3 "${base_dir}/cv_plot.py" \
        --cv "${cv_path}" \
        --pH "${pH}" \
        --tags-date "${file_date}" \
        --out-fig "${figures_dir}"
fi

if [[ "$action" == "all" || "$action" == "calc" ]]; then
    echo "=== [4] Running Electrochemical Calculation ==="
    python3 "${base_dir}/calculation.py" \
        --peaks "${data_dir}/Fc_SCE_${safe_electrode}_${file_date}_peaks.csv" \
        --calib-json "${calib_json_path}" \
        --tags-date "${file_date}" \
        --out-summary "${summary_dir}"
fi