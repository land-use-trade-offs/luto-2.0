#!/bin/bash

# Create a new Conda environment, and install the required packages
mamba create -c conda-forge -n luto python=3.11 -y
mamba activate luto


#  Install packages
mamba install -c conda-forge -y $(cat requirements_conda.txt)
pip install $(cat requirements_pip.txt)
