#!/bin/bash

# Valori di Autosubmit
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%     # YYYYMMDD
END_DATE=%CHUNK_END_DATE%
MEMBER=%MEMBER%

# Derivati
MEMBER_NUM="${MEMBER#fc}"
OUTDIR="${HPCROOTDIR}/DATA/${START_DATE}/${MEMBER}"

# ISO per i nomi file (uguale a quello usato nello script python)
START_ISO="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"

# Nomi file (coerenti con lo script Python)
SFC_GRIB="${OUTDIR}/output_${START_ISO}_r${MEMBER_NUM}_sfc.grib"
SFC_NC="${OUTDIR}/output_${START_ISO}_r${MEMBER_NUM}_sfc.nc"

# Interpolazione: target in Pa (hPa x 100)
cdo -f nc copy ${SFC_GRIB} ${SFC_NC}

# Remove the original GRIB file to save space
rm ${SFC_GRIB}
