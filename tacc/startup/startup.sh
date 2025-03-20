#!/bin/bash

module load gcc cuda openmpi phdf5 sqlite python3

# Store the random number in a variable for consistency
RAND_DIR=$SCRATCH/junk_envs/$RANDOM

mkdir -p "$RAND_DIR"
python3 -m venv "$RAND_DIR"
source "$RAND_DIR/bin/activate"

pip3 install h5py tqdm matplotlib seaborn scipy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -e /work/10407/anthony50102/vista/configmypy
python3 -m pip install -e /work/10407/anthony50102/vista/DQNO

# Jupyter stuff
pip3 install jupyter ipykernel
python3 -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
