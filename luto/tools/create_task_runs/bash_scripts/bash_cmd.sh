#!/bin/bash

#SBATCH -p mem                      	# Partition name
#SBATCH --time=30-24:00:00            	# Runtime in D-HH:MM:SS
#SBATCH --mem=100G                  	# Memory total in GB
#SBATCH --cpus-per-task=32          	# Number of CPUs per task
#SBATCH --job-name=luto                	# Job name



# Activate the Conda environment
conda activate luto

cd /home/jinzhu/LUTO2/luto_4_timeseries
python luto/run.sh