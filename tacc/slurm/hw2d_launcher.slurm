#!/bin/bash
#SBATCH   -p development
#SBATCH   -J launcher_hw2d
#SBATCH   -o slurm_outputs/launcher_hw2d.o%j
#SBATCH   -e slurm_outputs/launcher_hw2d.e%j
#SBATCH   -N 2
#SBATCH   -t 0:40:00

set -e

JOB_NAME="launcher_hw2d"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SCRATCH/venvs/${JOB_NAME}_${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

# Load necessary modules
module load python3/3.9.2
module load pylauncher
module load phdf5

# Create and activate virtual environment if not already present
VENV_PATH="${OUTPUT_DIR}/venv"
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip3 install --upgrade pip
    pip3 install hw2d[accelerators] numpy==1.26.4 paramiko h5py
else
    source "$VENV_PATH/bin/activate"
fi

python3 py_hw2d_launcher.py

# Clean up virtual environment after job completion
rm -rf "${OUTPUT_DIR}/venv"