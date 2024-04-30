#!/bin/bash

# Add the Conda binaries to the PATH
export PATH="$HOME/miniforge3/bin:$PATH"


# Install mini-forge
if [[ ! -d $HOME/miniforge3 ]]
then
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3
    rm Miniforge3-$(uname)-$(uname -m).sh
fi

# Initialize mamba or conda
mamba init

