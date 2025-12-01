#!/bin/bash
# ------------------------------------------------------------------
# Autosubmit job: run NeuralGCM inference (deterministic or stochastic)
# ------------------------------------------------------------------

set -euo pipefail

# === Autosubmit variables ===
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%   # YYYYMMDD
END_DATE=%CHUNK_END_DATE%
MEMBER=%MEMBER%
VARIABLES="%INFERENCE_RULES.VARIABLES%"


# === Derived variables ===
START_ISO="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"
MEMBER_NUM="${MEMBER#fc}"

INDIR="${HPCROOTDIR}/DATA/${START_DATE}"
INFILE="${INDIR}/${START_ISO}.nc"
FINAL_OUTDIR="${HPCROOTDIR}/INFERRED/"

N_STEPS=%INFERENCE_RULES.N_STEPS%

# === Model selection (set manually or via Autosubmit config) ===
MODEL_NAME=%INFERENCE_RULES.MODEL_NAME%
# Example alternatives:
# MODEL_NAME="v1/stochastic_1_4_deg.pkl"
# MODEL_NAME="v1_precip/stochastic_precip_2_8_deg.pkl"

# === Activate Python environment ===
source "$HPCROOTDIR/venv/bin/activate"
export JAX_PLATFORMS=cpu

# # === Choose which script to run based on model type ===
# if [[ "${MODEL_NAME}" == *"stochastic"* ]]; then
#     echo "Detected stochastic model → running infer-stoc.py"
#     SCRIPT="${HPCROOTDIR}/git_project/runscript/infer-stoc.py"
# else
#     echo "Detected deterministic model → running infer.py"
#     SCRIPT="${HPCROOTDIR}/git_project/runscript/infer.py"
# fi

SCRIPT="${HPCROOTDIR}/git_project/runscript/infer.py"

# === Run inference ===
python "${SCRIPT}" \
  --input_path "${INFILE}" \
  --output_path "${FINAL_OUTDIR}" \
  --num_steps "${N_STEPS}" \
  --model_name "${MODEL_NAME}" \
  --member "${MEMBER_NUM}" \
  --variables ${VARIABLES// /,}


# Pressure levels (hPa)
PLEVELS="1,5,10,20,30,50,70,100,150,200,250,300,400,500,600,700,850,925,1000"

for member in ${MEMBERS}; do
  for variable in ${VARIABLES}; do

    in_file="${FINAL_OUTDIR}/${variable}/fc${member}/${initial_time}_${variable}_r${member}_infer.nc"
    out_file="${in_file/_infer/}"

    mkdir -p "$(dirname "${out_file}")"

    # Select requested pressure levels
    cdo -s sellevel,${PLEVELS} "${in_file}" "${out_file}"

    # Remove intermediate *_infer file
    rm -f "${in_file}"
  done
done