#!/bin/bash
# PREPARE-ICS-CMIP6
set -xuve

# Valori riempiti da Autosubmit:
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%     # formati tipici: YYYYMMDD
END_DATE=%CHUNK_END_DATE%         # stesso formato dei chunk
GIT_ORIGIN=%GIT.PROJECT_ORIGIN%
PROJECT_BRANCH=%GIT.PROJECT_BRANCH%

cd $HPCROOTDIR
git clone $GIT_ORIGIN -b $PROJECT_BRANCH git_project