#!/bin/bash



# Create a new Conda environment, and install the required packages
mamba create -c conda-forge -n luto_py311 python=3.11 -y
mamba activate luto_py311


#  Install packages
mamba install -c conda-forge -y $(cat conda_pkg.txt)
pip install $(cat pip_pkg.txt)
