# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
param
(
    [string]$ArtifactsDirectory = $env:SYSTEM_ARTIFACTSDIRECTORY,
    [Parameter(Mandatory)][string]$TestArtifactPath,
    [Parameter(Mandatory)][string]$TensorFlowPackage
)

$ErrorActionPreference = 'Stop'

$InstallDir = Join-Path ($ArtifactsDirectory | Resolve-Path) "miniconda3"
$PluginPackage = (Get-ChildItem "$TestArtifactPath/tensorflow_directml_plugin*.whl").FullName
$TestEnvPath = "$ArtifactsDirectory/test_env"
$TestArtifact = $TestArtifactPath | Split-Path -Leaf
$PyVersionMajorDotMinor = $TestArtifact -replace '.*-cp(\d)(\d)', '$1.$2'

Write-Host "Installing miniconda3 to $InstallDir"
$Url = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe'
$DownloadPath = "$ArtifactsDirectory/miniconda.exe"
(New-Object System.Net.WebClient).DownloadFile($Url, $DownloadPath)
Start-Process $DownloadPath -ArgumentList '/NoRegistry=1', '/InstallationType=JustMe', '/RegisterPython=0', '/S', "/D=$InstallDir" -Wait
& "$InstallDir/shell/condabin/conda-hook.ps1"

conda create --prefix $TestEnvPath python=$PyVersionMajorDotMinor -y
conda activate $TestEnvPath
pip install $TensorFlowPackage
# pip install tensorboard_plugin_profile
pip install $PluginPackage
pip list

$ActivateCmd = "$InstallDir/shell/condabin/conda-hook.ps1; conda activate $TestEnvPath"
echo "##vso[task.setVariable variable=activateCommand;isOutput=true]$ActivateCmd"