#!/bin/bash

# Argument parser using getopts
usage() {
    echo "Usage: $0 -a <action: plot|calc> -e <electrode> -n <experiment number> -s <scan rate> -c <condition> -d <date> -p <pH>"
    echo "  -e   Electrode string (e.g. \"Au, Pt small, SCE\")"
    echo "  -n   Experiment number (e.g. 1)"
    echo "  -s   Scan rate (e.g. 10mV)"
    echo "  -c   Condition (e.g. normal)"
    echo "  -d   Date (e.g. 2-10-25)"
    echo "  -p   pH value (e.g. 7.4)"
    echo ""
    echo "Note: Potential shift is now automatically loaded from the calibration JSON file."
    echo "Example: $0 -a plot -e \"Au, Pt small, SCE\" -n 1 -s 10mV -c normal -d 2-10-25 -p 7.4"
    exit 1
}

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

# Check required arguments
if [ -z "$action" ] || [ -z "$electrode" ] || [ -z "$no_exp" ] || [ -z "$scan_rate" ] || [ -z "$condition" ] || [ -z "$file_date" ] || [ -z "$pH" ]; then
    usage
fi

# Paths
base_dir="/Users/tai/Downloads/Research/PDA-based memristors/plotting"
data_dir="${base_dir}/data"
graphics_dir="${base_dir}/graphics"

# Path generation
sce_calib_path="${data_dir}/Fc_SCE_${electrode}_${file_date}.txt"
pretreat_path="${data_dir}/Pre_treat_${no_exp}_${electrode}_${condition}_${file_date}.txt"
cv_path="${data_dir}/PDA_${no_exp}_${electrode}_${scan_rate}_${file_date}.txt"

# Calibration JSON path for calculation
calib_json_path="${data_dir}/RHE_Fc_SCE_${electrode}_${file_date}.json"

if [ "$action" = "plot" ]; then
    echo "=== Running CV plotting for ${file_date} ==="
    python3 "${base_dir}/cv_plot.py" --sce "${sce_calib_path}" --cv "${cv_path}" --pretreat "${pretreat_path}"  --pH "${pH}" --tags-date "${file_date}"
elif [ "$action" = "calc" ]; then
    echo "=== Running electrochemical calculation for ${file_date} ==="
    python3 "${base_dir}/calculation.py" \
        --peaks "${data_dir}/Fc_SCE_${electrode}_${file_date}_peaks.csv" \
        --tags-date "${file_date}" \
        --calib-json "${calib_json_path}"
else
    usage
fi