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
PL_GRIB="${OUTDIR}/output_${START_ISO}_r${MEMBER_NUM}_pl.grib"
PL_NC="${OUTDIR}/output_${START_ISO}_r${MEMBER_NUM}_pl.nc"
PL_INT_NC="${OUTDIR}/output_${START_ISO}_r${MEMBER_NUM}_pl_interpolated.nc"

# Interpolazione: target in Pa (hPa x 100)
cdo -f nc copy ${PL_GRIB} ${PL_NC}

cdo -f nc4c -z zip_6 -b F32 intlevel,100,200,300,500,700,1000,2000,3000,5000,7000,10000,12500,15000,17500,20000,22500,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,77500,80000,82500,85000,87500,90000,92500,95000,97500,100000 ${PL_NC} ${PL_INT_NC}

