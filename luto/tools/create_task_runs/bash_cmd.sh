#!/bin/bash

#SBATCH -p mem                      	# Partition name
#SBATCH --time=30-24:00:00            	# Runtime in D-HH:MM:SS
#SBATCH --mem=100G                  	# Memory total in GB
#SBATCH --cpus-per-task=32          	# Number of CPUs per task
#SBATCH --job-name=luto                	# Job name


# Install conda
if ! command -v conda &> /dev/null
then
    # Download miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
    # Install miniconda
    bash $HOME/miniconda.sh -b -p $HOME/miniconda
    # Remove the installer
    rm $HOME/miniconda.sh
fi



# Install mini-forge
if [[ $PATH != *miniforge3* ]]
then
    # Install mini-forge
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3

    # Delete the script
    rm Miniforge3-$(uname)-$(uname -m).sh
fi




# Initialize Conda
source $HOME/miniforge3/etc/profile.d/conda.sh

# Create a new Conda environment, and install the required packages
mamba create -c conda-forge -n luto_py311 python=3.11 -y
conda activate luto_py311
mamba install -c conda-forge -n luto_py311 -y --file requirements.txt




