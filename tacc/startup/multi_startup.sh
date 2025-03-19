#!/bin/bash

module load gcc cuda openmpi
module load python3_mpi
module load phdf5
module load sqlite

# Store the random number in a variable for consistency
RAND_DIR=$SCRATCH/junk_envs/$RANDOM

mkdir -p "$RAND_DIR"
python3 -m venv "$RAND_DIR"
source "$RAND_DIR/bin/activate"

pip3 install h5py tqdm matplotlib seaborn scipy
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
python3 -m pip install -e /work/10407/anthony50102/vista/configmypy
python3 -m pip install -e /work/10407/anthony50102/vista/DQFNO

# Jupyter stuff
pip3 install jupyter ipykernel
python3 -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
# TODO - fix this
# export PYTHONPATH="$RAND_DIR/lib/python3.9/site-packages"
