#!/bin/bash
# PREPARE-ICS-CMIP6
set -xuve

# Valori riempiti da Autosubmit:
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%     # formati tipici: YYYYMMDD
END_DATE=%CHUNK_END_DATE%         # stesso formato dei chunk


# Dove salvare i dati
OUTDIR="${HPCROOTDIR}/DATA/${START_DATE}"
mkdir -p "${OUTDIR}"

# Config per lo script Python
CONFIG_DIR="${HPCROOTDIR}/conf/${START_DATE}"
mkdir -p "${CONFIG_DIR}"
CONFIG_FILE="${CONFIG_DIR}/${START_DATE}.yaml"

# Converti YYYYMMDD -> YYYY-MM-DD (GNU date)
START_ISO="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"
END_ISO="${END_DATE:0:4}-${END_DATE:4:2}-${END_DATE:6:2}"

# Scrivi il config
cat > "${CONFIG_FILE}" <<EOF
start_time: ${START_ISO}
end_time: ${END_ISO}
output_path: ${OUTDIR}
EOF

# Ambiente (usa il tuo venv)
source /home/jgrassi/code/mars/venv/bin/activate

# Esegui lo script Python
python ${PROJDIR}/runscript/download-EERIE.py --config "${CONFIG_FILE}"