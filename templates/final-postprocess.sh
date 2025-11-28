#!/usr/bin/env bash
set -euo pipefail

# === Autosubmit variables ===
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%   # YYYYMMDD
END_DATE=%CHUNK_END_DATE%
MEMBERS="%EXPERIMENT.MEMBER%"
VARIABLES="%INFERENCE_RULES.VARIABLES%"

START_ISO="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"
FINAL_OUTDIR="${HPCROOTDIR}/INFERRED/"

# Pressure levels (hPa)
PLEVELS="1,5,10,20,30,50,70,100,150,200,250,300,400,500,600,700,850,925,1000"

# Use START_DATE as initial_time in filenames (adjust if needed)
initial_time="${START_DATE}"

for member in ${MEMBERS}; do
  for variable in ${VARIABLES}; do

    in_file="${FINAL_OUTDIR}/${member}/${initial_time}_${variable}_r${member}_infer.nc"
    out_file="${in_file/_infer/}"

    mkdir -p "$(dirname "${out_file}")"

    # Select requested pressure levels
    cdo -s sellevel,${PLEVELS} "${in_file}" "${out_file}"

    # Remove intermediate *_infer file
    rm -f "${in_file}"
  done
done