#!/bin/bash
#SBATCH -J train_DQFNO           # Job name
#SBATCH -o slurm_outputs/train_DQFNO.o%j            # Name of stdout output file
#SBATCH -e slurm_outputs/train_DQFNO.e%j            # Name of stderr error file
#SBATCH -p gh                # Queue (partition) name
#SBATCH -N 1                    # Total # of nodes (must be 1 for serial)
#SBATCH -t 10:00:00             # Run time (hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH -A $PARTITION
#SBATCH --mail-user=$EMAIL

set -e

#Load startup script
source /home1/10407/anthony50102/startup.sh

cd /work/10407/anthony50102/vista/DQFNO

python3 ../../scripts/train.py "$@"