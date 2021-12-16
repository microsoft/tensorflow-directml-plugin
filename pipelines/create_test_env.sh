#!/usr/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

test_artifact_path=$1
tensorflow_package=$2

# Windows agents use the agent artifacts directory for the conda installation, but
# this is slow in WSL (filesystem networking overhead). Instead the agent will use
# a temp directory in the native Linux filesystem.
tmp_testing_root="/tmp/tfdml_plugin_pipeline"
rm -rf "$tmp_testing_root"
mkdir "$tmp_testing_root"
cd $tmp_testing_root

install_dir="$tmp_testing_root/miniconda3"
plugin_package=$(ls $test_artifact_path/tensorflow_directml_plugin*.whl)
test_env_path="$tmp_testing_root/test_env"
test_artifact=$(basename $test_artifact_path)
py_version_major_dot_minor=$(echo $test_artifact | sed -E "s/.*-cp([0-9])([0-9])/\1.\2/")

echo "Installing miniconda3 to $install_dir"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $install_dir
eval "$($install_dir/bin/conda shell.bash hook)" 
conda create --prefix $test_env_path python=$py_version_major_dot_minor -y

conda activate $test_env_path
pip install $tensorflow_package
pip install tensorboard_plugin_profile
pip install $plugin_package
pip list

activate_cmd="source $install_dir/bin/activate $test_env_path"
echo "##vso[task.setVariable variable=activateCommand;isOutput=true]$activate_cmd"