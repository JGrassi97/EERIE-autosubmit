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


# === Derived variables ===
START_ISO="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"
MEMBER_NUM="${MEMBER#fc}"

INDIR="${HPCROOTDIR}/DATA/${START_DATE}"
INFILE="${INDIR}/${START_ISO}.nc"
FINAL_OUTDIR="${HPCROOTDIR}/INFERRED/${MEMBER}"
mkdir -p "${FINAL_OUTDIR}"
FINAL_FILE="${FINAL_OUTDIR}/${START_ISO}_r${MEMBER_NUM}_infer.nc"

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
  --output_path "${FINAL_FILE}" \
  --num_steps "${N_STEPS}" \
  --model_name "${MODEL_NAME}" \
  --member "${MEMBER_NUM}"

# === Optional: clean up input file after processing ===
rm -f "${INFILE}"