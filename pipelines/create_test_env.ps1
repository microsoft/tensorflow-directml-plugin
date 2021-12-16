# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
param
(
    [string]$StagingDirectory = $env:BUILD_STAGINGDIRECTORY,
    [Parameter(Mandatory)][string]$ArtifactPath,
    [Parameter(Mandatory)][string]$TensorFlowPackage
)

$ErrorActionPreference = 'Stop'

$InstallDir = Join-Path ($StagingDirectory | Resolve-Path) "miniconda3"
$PluginPackage = (Get-ChildItem "$ArtifactPath/tensorflow_directml_plugin*.whl").FullName

Write-Host "Installing miniconda3 to $InstallDir"
$Url = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe'
$DownloadPath = "$StagingDirectory/miniconda.exe"
(New-Object System.Net.WebClient).DownloadFile($Url, $DownloadPath)
Start-Process $DownloadPath -ArgumentList '/NoRegistry=1', '/InstallationType=JustMe', '/RegisterPython=0', '/S', "/D=$InstallDir" -Wait
& "$InstallDir/shell/condabin/conda-hook.ps1"

$Artifact = $ArtifactPath | Split-Path -Leaf

$PyVersionMajorDotMinor = $Artifact -replace '.*-cp(\d)(\d)', '$1.$2'
$TestEnvPath = "$StagingDirectory/test_env"
conda create --prefix $TestEnvPath python=$PyVersionMajorDotMinor -y
conda activate $TestEnvPath
pip install $TensorFlowPackage
pip install $PluginPackage
pip list

$ActivateCmd = "$InstallDir/shell/condabin/conda-hook.ps1; conda activate $TestEnvPath"
echo "##vso[task.setVariable variable=activateCommand;isOutput=true]$ActivateCmd"