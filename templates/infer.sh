# Valori di Autosubmit
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%     # YYYYMMDD
END_DATE=%CHUNK_END_DATE%
MEMBER=%MEMBER%


START_ISO="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"
MEMBER_NUM="${MEMBER#fc}"

INDIR="${HPCROOTDIR}/DATA/${START_DATE}/${MEMBER}"
INFILE="${INDIR}/${START_ISO}_r${MEMBER_NUM}.nc"

OUTDIR="${HPCROOTDIR}/DATA/${START_DATE}/${MEMBER}/inference"
mkdir -p "${OUTDIR}"
OUTFILE="${OUTDIR}/${START_ISO}_r${MEMBER_NUM}.nc"

# Activate Python virtual environment
source /home/jgrassi/code/neuralGCM/venv/bin/activate

python infer.py \
  --input_path "${INFILE}" \
  --output_path "${OUTFILE}" \