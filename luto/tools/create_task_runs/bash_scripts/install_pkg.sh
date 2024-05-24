#!/bin/bash

# Add the Conda binaries to the PATH
source ~/.bashrc

py_ver='python=3.12'
py_ver_=$(echo $py_ver | sed 's/=//g')

# Create a new Conda environment, and install the required packages
mamba create -c conda-forge -n luto ${py_ver} -y
mamba activate luto


# Install packages using conda in the "luto" environment
mamba install -n luto -c conda-forge -y $(cat requirements_conda.txt)

# Install packages using pip in the "luto" environment
pip install --target=$HOME/miniforge3/envs/luto/lib/${py_ver_}/site-packages $(cat requirements_pip.txt)
