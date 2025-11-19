#!/bin/bash
# PREPARE-ICS-CMIP6
set -xuve

# Values filled automatically by Autosubmit:
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%     # Typical format: YYYYMMDD
END_DATE=%CHUNK_END_DATE%         # Same format as the chunk
MEMBER=%MEMBER%


# Configuration directory for the Python script
CONFIG_DIR="${HPCROOTDIR}/conf/${START_DATE}/${MEMBER}"
CONFIG_FILE="${CONFIG_DIR}/${MEMBER}.yaml"

# Activate Python virtual environment
source /home/jgrassi/code/mars/venv/bin/activate

# Run the Python script
python ${HPCROOTDIR}/git_project/runscript/download-EERIE-pl.py --config "${CONFIG_FILE}"