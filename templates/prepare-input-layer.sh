# Valori di Autosubmit
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%     # YYYYMMDD
END_DATE=%CHUNK_END_DATE%
MEMBER=%MEMBER%

OUTDIR="${HPCROOTDIR}/DATA/${START_DATE}/${MEMBER}"
START_ISO="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"
MEMBER_NUM="${MEMBER#fc}"

SFC_NC="${OUTDIR}/output_${START_ISO}_r${MEMBER_NUM}_sfc.nc"
PL_INT_NC="${OUTDIR}/output_${START_ISO}_r${MEMBER_NUM}_pl_interpolated.nc"
OUTFILE="${OUTDIR}/${START_ISO}_r${MEMBER_NUM}.nc"

# Activate Python virtual environment
source /home/jgrassi/code/neuralGCM/venv/bin/activate

# === Run the Python preparation script ===
python "${HPCROOTDIR}/git_project/runscript/prepare_input_layer.py" \
  --pl_path "${PL_INT_NC}" \
  --sfc_path "${SFC_NC}" \
  --out_path "${OUTFILE}"


rm "${PL_INT_NC}"
rm "${SFC_NC}"