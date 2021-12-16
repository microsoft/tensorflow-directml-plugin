#!/usr/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

miniconda_path="miniconda3"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $miniconda_path
eval "$($miniconda_path/bin/conda shell.bash hook)" 
conda create --name build python=$(vars.pyVersionMajorDotMinor) -y
activate_cmd="source $(Build.StagingDirectory)/$miniconda_path/bin/activate build"
echo "##vso[task.setVariable variable=activateCommand;isOutput=true]$activate_cmd"