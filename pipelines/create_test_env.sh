#!/usr/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

artifacts_path=$1
test_artifact_path=$2
tensorflow_package=$3

install_dir="$artifacts_directory/miniconda3"
plugin_package=$(ls $test_artifact_path/tensorflow_directml_plugin*.whl)
test_env_path="$artifacts_directory/test_env"
test_artifact=$(basename $test_artifact_path)
py_version_major_dot_minor=$(echo $test_artifact | sed -E "s/.*-cp([0-9])([0-9])/\1.\2/")

echo "Installing miniconda3 to $install_dir"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $install_dir
eval "$($install_dir/bin/conda shell.bash hook)" 
conda create --prefix $test_env_path python=$(vars.pyVersionMajorDotMinor) -y

activate_cmd="source $install_dir/bin/activate $test_env_path"
echo "##vso[task.setVariable variable=activateCommand;isOutput=true]$activate_cmd"