#!/bin/bash
# PREPARE-ICS-CMIP6
set -xuve

# Values filled automatically by Autosubmit:
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%     # Typical format: YYYYMMDD
END_DATE=%CHUNK_END_DATE%         # Same format as the chunk
MEMBER=%MEMBER%

MEMBER_NUM="${MEMBER#fc}"  # Remove "fc" from the beginning

# Where to store the output data
OUTDIR="${HPCROOTDIR}/DATA/${START_DATE}"
mkdir -p "${OUTDIR}"

# Configuration directory for the Python script
CONFIG_DIR="${HPCROOTDIR}/conf/${START_DATE}"
mkdir -p "${CONFIG_DIR}"
CONFIG_FILE="${CONFIG_DIR}/config.yaml"

# Convert YYYYMMDD → YYYY-MM-DD
START_ISO="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"
END_ISO="${END_DATE:0:4}-${END_DATE:4:2}-${END_DATE:6:2}"

# Write configuration file in YAML format
cat > "${CONFIG_FILE}" <<EOF
start_time: ${START_ISO}
end_time: ${END_ISO}
output_path: ${OUTDIR}
EOF