#!/bin/bash
# PREPARE-ICS-CMIP6
set -xuve

# -----------------------------
# Autofilled by Autosubmit
# -----------------------------
HPCROOTDIR=%HPCROOTDIR%
PROJDIR=%PROJDIR%
START_DATE=%CHUNK_START_DATE%
END_DATE=%CHUNK_END_DATE%
GIT_ORIGIN=%GIT.PROJECT_ORIGIN%
PROJECT_BRANCH=%GIT.PROJECT_BRANCH%

# -----------------------------
# Create project directory
# -----------------------------
mkdir -p "$HPCROOTDIR"
cd "$HPCROOTDIR"

# -----------------------------
# Clone the git repository
# -----------------------------
if [ ! -d git_project ]; then
    git clone "$GIT_ORIGIN" -b "$PROJECT_BRANCH" git_project
else
    echo "git_project already exists → skipping clone"
fi

# -----------------------------
# Prepare Python venv via uv
# -----------------------------
# Install uv locally if missing
if ! command -v uv &>/dev/null; then
    echo "uv not found → installing locally"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure Python 3.11 is available under uv
uv python install 3.11

# Create venv inside HPCROOTDIR (idempotent)
if [ ! -d "$HPCROOTDIR/venv" ]; then
    uv venv --python "$(uv python find 3.11)" "$HPCROOTDIR/venv"
else
    echo "venv already exists → skipping creation"
fi

# Activate venv
source "$HPCROOTDIR/venv/bin/activate"

# -----------------------------
# Install runtime dependencies
# -----------------------------
# Use python -m pip to avoid interference with user/global pip
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    ecmwf-api-client \
    "xarray[complete]" \
    numpy \
    neuralgcm \
    gcsfs