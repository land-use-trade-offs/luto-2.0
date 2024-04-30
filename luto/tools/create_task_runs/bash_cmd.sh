#!/bin/bash

#SBATCH -p dgx                      		# Partition name
#SBATCH --time=24:00:00            	# Runtime in D-HH:MM:SS
#SBATCH --mem=100G                  	# Memory total in GB
#SBATCH --cpus-per-task=32          	# Number of CPUs per task
#SBATCH --job-name=luto                	# Job name


# Initialize Conda
source /home/jinzhu/miniforge3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate luto_py311

cd /home/jinzhu/LUTO2/luto_4_timeseries
python luto/run.sh




