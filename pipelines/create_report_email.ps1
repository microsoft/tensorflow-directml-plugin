# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Creates and emails an HTML build pipeline report.
#>
param
(
    [Parameter(Mandatory)][string]$TestArtifactsPath,
    [Parameter(Mandatory)][string]$BuildArtifactsPath,
    [Parameter(Mandatory)][string]$PipelineRunID,
    [Parameter(Mandatory)][string]$AccessToken,
    [Parameter(Mandatory)][string]$OutputHtmlPath,
    [string]$EmailTo
)

# TODO: REMOVE
$env:SYSTEM_COLLECTIONURI = "https://dev.azure.com/microsoft/"
$env:SYSTEM_TEAMPROJECT = "WindowsAI"

if (!$AccessToken)
{
    throw "This script requires a personal access token to use the REST API."
}

$Green = '#C0FFC0'
$Red = '#FFC0C0'
$Yellow = '#FFFFAA'
$Gray = '#DDDDDD'
$LightGray = '#E6E6E6'

# ---------------------------------------------------------------------------------------------------------------------
# Parse Artifacts
# ---------------------------------------------------------------------------------------------------------------------

."$PSScriptRoot\ADOHelper.ps1"
$Ado = [ADOHelper]::CreateFromPipeline($AccessToken)
$Run = $Ado.GetBuild($PipelineRunID)

$PipelineDuration = (Get-Date) - [datetime]$Run.StartTime

# e.g. refs/heads/master -> master
$ShortBranchName = $Run.SourceBranch -replace '^refs/heads/'

$RunUrl = "$env:SYSTEM_COLLECTIONURI$env:SYSTEM_TEAMPROJECT/_build/results?buildId=$($PipelineRunID)"

# Only get commits associated with this build if there was at least one earlier successful build of this branch.
$FirstRunOnBranch = $Ado.InvokeProjectApi("build/builds?definitions=$($Run.definition.id)&branchName=$($Run.SourceBranch)&`$top=1&resultFilter=succeeded&api-version=5.0", "GET", $null).Count -eq 0
if (!$FirstRunOnBranch)
{
    $Commits = $Ado.InvokeProjectApi("build/builds/${PipelineRunID}/changes?api-version=5.0", 'GET', $null).Value
}

$SucceededBuildCount = 0
$FailedBuildCount = 0
$OtherBuildCount = 0
$BuildMetadata = @{}
$MetadataItems = Get-ChildItem $BuildArtifactsPath -Filter '*.json'
foreach ($MetadataItem in $MetadataItems)
{
    $ArtifactMetadata = Get-Content $MetadataItem.FullName -Raw | ConvertFrom-Json
    $BuildMetadata[$MetadataItem.BaseName] = $ArtifactMetadata
    switch ($ArtifactMetadata.Status)
    {
        'Succeeded' { $SucceededBuildCount++ }
        'Failed' { $FailedBuildCount++ }
        default { $OtherBuildCount++ }
    }
}

if ($FailedBuildCount -gt 0)
{
    $BuildResult = 'Failed'
}
elseif ($SucceededBuildCount -eq $MetadataItems.Count)
{
    $BuildResult = 'Succeeded'
}
else
{
    $BuildResult = 'Partially Succeeded'
}

$TestsArtifactsExist = Get-ChildItem $TestArtifactsPath -ErrorAction Ignore
if ($TestsArtifactsExist)
{
    $TestSummary = Get-Content "$TestArtifactsPath/test_summary.json" -Raw | ConvertFrom-Json
    $AgentSummary = Get-Content "$TestArtifactsPath/agent_summary.json" -Raw | ConvertFrom-Json

    $TestGroups = ($TestSummary | Get-Member -MemberType NoteProperty).Name
    $HasPassedTests = ($TestGroups | % { $TestSummary.$_.tests_passed_count } | Measure-Object -sum).sum -gt 0
    $HasFailures = ($TestGroups | % { $TestSummary.$_.tests_failed_count } | Measure-Object -sum).sum -gt 0
    
    if ($HasFailures)
    {
        $TestRunResult = 'Failed'
    }
    elseif ($HasPassedTests)
    {
        $TestRunResult = 'Skipped'
    }
    else
    {
        $TestRunResult = 'Succeeded'
    }
}

# ---------------------------------------------------------------------------------------------------------------------
# Create HTML
# ---------------------------------------------------------------------------------------------------------------------

$Html = [System.Collections.ArrayList]::new()

# ---------------------------------------------------------------------------------------------------------------------
# Summary Table
#
# Example:
#
# | ---------------------|--------|------------------------------------------|----------|------------------|
# | Pipeline             | Branch | Version                                  | Reason   | Duration         |
# | ---------------------|--------|------------------------------------------|----------|------------------|
# | 201123-2300.1.master | master | fda4ad7726f5d80d8cf21fb06b06d58a673bb9bc | schedule | 01:58:44.4982976 |
# | ---------------------|--------|------------------------------------------|----------|------------------|
#
# ---------------------------------------------------------------------------------------------------------------------

$Headers = 'Pipeline', 'Branch', 'Version', 'Reason', 'Duration'
$Style = "padding:1px 3px; border:1px solid gray; border-left:1px solid gray"

$Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%;`">"
$Html += "<tr>"
foreach ($Header in $Headers)
{
    $Html += "<th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`">$Header</th>"
}
$Html += "</tr>"
$Html += '<tr>'
$Html += "<td style=`"$Style`"><a target=`"_blank`" href=`"$RunUrl`">$($Run.BuildNumber)</a></td>"
$Html += "<td style=`"$Style`">$($Run.SourceBranch)</td>"
$Html += "<td style=`"$Style`">$($Run.SourceVersion)</td>"
$Html += "<td style=`"$Style`">$($Run.Reason)</td>"
$Html += "<td style=`"$Style`">$($PipelineDuration.ToString("c"))</td>"
$Html += '</tr>'
$Html += "</table><br>"

# ---------------------------------------------------------------------------------------------------------------------
# Commits Table
#
# Example:
#
# |----------------------------------------------------------------------------------------|  
# |                                       Commits                                          |
# |----------------------------------------------------------------------------------------|                                         
# | fda4ad77 | 2020-11-23 20:29:51 | alpha@microsoft.com : Merged PR 3: Fix the build      |
# | 7ffa8445 | 2020-11-23 20:29:51 | bravo@microsoft.com : Merged PR 2: Added bar          |
# | e00e2a5e | 2020-11-23 20:29:51 | charlie@microsoft.com : Merged PR 1: Added foo        |
# |----------|---------------------|-------------------------------------------------------|  
#
# ---------------------------------------------------------------------------------------------------------------------

if ($Commits.Count -gt 0)
{
    $Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%;`">"
    $Html += "<tr>"
    $Html += "<th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`" colspan=3>Commits</th>"
    $Html += "</tr>"

    foreach ($Commit in $Commits)
    {
        $Url = "https://github.com/microsoft/tensorflow-directml-plugin/commit/$($Commit.id)"
        $Timestamp = ([datetime]$Commit.timestamp).ToString("yyyy-MM-dd HH:mm:ss")

        $Style = "padding:1px 3px; border-bottom:1px solid gray; border-left:1px solid gray"
        $Html += "<tr style=`"text-align:left;`">"
        $Html += "<td style=`"$Style; font-family:monospace;`"><a target=`"_blank`" href=`"$($Url)`">$($Commit.id.substring(0,8))</a></td>"
        $Html += "<td style=`"$Style;`">$Timestamp</td>"
        $Html += "<td style=`"$Style; border-right:1px solid gray;`"><b>$($Commit.author.displayName)</b> : $($Commit.message)</td>"
        $Html += "</tr>"
    }

    $Html += "</table><br>"
}

# ---------------------------------------------------------------------------------------------------------------------
# Build Banner
#
# Example:
#
# | -----------------|
# | Build SUCCEEDED  |
# | -----------------|
#
# ---------------------------------------------------------------------------------------------------------------------

switch ($BuildResult)
{
    'Succeeded' { $Color = $Green }
    'Failed' { $Color = $Red }
    default { $Color = $Yellow }
}

$Html += "<h1 style=`"text-align:center; border:1px solid gray; background:$Color`">Build $($BuildResult.ToUpper())</h1>"

# ---------------------------------------------------------------------------------------------------------------------
# Build Table
#
# Example:
#
# |---------------------------|-----------|------------|------------------|
# | Name                      | Status    | Agent      | Duration         |
# |---------------------------|-----------|------------|------------------|
# | arm64-linux-inbox-debug   | Succeeded | AP-BUILD02 | 00:13:22.4982976 |
# | arm64-linux-inbox-release | Succeeded | AP-BUILD01 | 00:12:23.8966295 |
# | x64-win-inbox-debug       | Failed    | AP-BUILD03 | 00:17:12.5922936 |
# |---------------------------|-----------|------------|------------------|
#
# ---------------------------------------------------------------------------------------------------------------------

$Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%;`">"
$Html += "<tr>"
$Headers = 'Name', 'Status', 'Agent', 'Duration'
foreach ($Header in $Headers)
{
    $Html += "<th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`">$Header</th>"
}
$Html += "</tr>"

foreach ($BuildName in ($BuildMetadata.Keys | Sort-Object))
{
    $Metadata = $BuildMetadata[$BuildName]
    switch ($Metadata.status)
    {
        'Succeeded' { $Color = $Green }
        'SucceededWithIssues' { $Color = $Yellow }
        'Failed' { $Color = $Red }
        default { $Color = $Gray }
    }

    $Style = "padding:1px 3px; border-bottom:1px solid gray; border-left:1px solid gray; background-color: $Color"
    $Html += "<tr style=`"text-align:left;`">"
    $Html += "<td style=`"$Style;`">$BuildName</td>"
    $Html += "<td style=`"$Style;`">$($Metadata.status)</td>"
    $Html += "<td style=`"$Style;`">$($Metadata.agentName)</td>"
    $Html += "<td style=`"$Style; border-right:1px solid gray;`">$($Metadata.duration)</td>"
    $Html += "</tr>"
}

$Html += "</table><br>"

$Html | Out-File $OutputHtmlPath -Encoding utf8
Write-Host "##vso[task.uploadsummary]$(Resolve-Path $OutputHtmlPath)"

# ---------------------------------------------------------------------------------------------------------------------
# Test Results Banner
#
# Example:
#
# | -----------------|
# | Tests SUCCEEDED  |
# | -----------------|
#
# ---------------------------------------------------------------------------------------------------------------------

if ($TestsArtifactsExist)
{
    switch ($TestRunResult)
    {
        'Succeeded' { $Color = $Green }
        'Failed' { $Color = $Red }
        default { $Color = $Yellow }
    }

    $Html += "<h1 style=`"text-align:center; border:1px solid gray; background:$Color`">Tests $($TestRunResult.ToUpper())</h1>"
}

# ---------------------------------------------------------------------------------------------------------------------
# Test Results
# ---------------------------------------------------------------------------------------------------------------------

if ($TestsArtifactsExist)
{
    $Headers = 
        'Group',
        'Agent',
        'System' ,
        'Build',
        'Total',
        'Passed',
        'Failed',
        'Skipped',
        'Timed Out',
        'Time'

    $HeadersStyle = "border: 1px solid gray; background-color: white; color:black"
    $HeaderTags = $Headers | ForEach-Object { "<th style=`"$HeadersStyle`">$_</th>" }

    $Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%`">"
    $Html += "<tr>$HeaderTags</tr>"

    foreach ($TestGroup in $TestGroups)
    {
        $TestGroupSummary = $TestSummary.$TestGroup
        $TestGroupArtifactCount = $TestGroupSummary.Count
        $FirstGroupRow = $True

        # Determine group cell color.
        if     ($TestGroupSummary.tests_failed_count -gt 0) { $GroupColor = $Red }
        elseif ($TestGroupSummary.tests_passed_count -gt 0) { $GroupColor = $Green }
        else                                                { $GroupColor = $Gray }

        $AgentJobs = $TestGroupSummary | Sort-Object -Property agent
        $CurrentAgentName = $null
        foreach ($AgentJob in $AgentJobs)
        {
            $AgentName = $AgentJob.agent
            $FirstAgentRow = $CurrentAgentName -ne $AgentName
            $CurrentAgentName = $AgentName
            $AgentArtifactsCount = ($AgentJobs | ? agent -eq $AgentName).count

            $AgentInfo = $AgentSummary.$AgentName

            # Determine agent cell color.
            if     ($AgentJob.tests_failed_count -gt 0) { $AgentColor = $Red }
            elseif ($AgentJob.tests_passed_count -gt 0) { $AgentColor = $Green }
            else                                        { $AgentColor = $Gray }

            if ($AgentJob.agentHasResults)
            {
                $Time = $AgentJob.time_seconds
            }
            else
            {
                if (!$AgentJob.agentWasEnabled)
                {
                    $Time = 'DISABLED'
                }
                elseif (!$AgentJob.agentWasOnline)
                {
                    $Time = 'OFFLINE'
                }
                else
                {
                    $Time = $null
                }
            }

            $Html += "<tr>"

            # Determine result cell color.
            if     ($AgentJob.tests_failed_count)       { $ResultColor = $Red }
            elseif ($AgentJob.tests_passed_count -gt 0) { $ResultColor = $Green }
            else                                  { $ResultColor = $Gray }

            if ($FirstGroupRow)
            {
                $FirstGroupRow = $False
                $CellStyle = "border:1px solid gray; background-color: $GroupColor"
                $Html += "<td rowspan=`"$TestGroupArtifactCount`" colspan=`"1`" style=`"$CellStyle`">$TestGroup</td>"
            }

            $CellStyle = "border:1px solid gray; background-color: $AgentColor"
            if ($FirstAgentRow)
            {
                $FirstAgentRow = $False

                $Html += "<td rowspan=`"$($AgentArtifactsCount)`" style=`"$CellStyle`">$($AgentName)</td>"

                if ($AgentInfo)
                {
                    $SystemInfo = "$($AgentInfo.DisplayAdapter)<br>$($AgentInfo.DisplayDriver)"
                }
                else
                {
                    if (!$AgentJob.agentWasEnabled)
                    {
                        $SystemInfo = "DISABLED"
                    }
                    else
                    {
                        $SystemInfo = "UNKNOWN"
                    }
                }

                $Html += "<td rowspan=`"$($AgentArtifactsCount)`" style=`"$CellStyle; text-align: left`">$SystemInfo</td>"
            }

            $CellStyle = "border:1px solid gray; background-color: $ResultColor"
    
            $Html += "<td style=`"$CellStyle`">$($AgentJob.build)</td>"
            $Html += "<td style=`"$CellStyle`">$($AgentJob.tests_total_count)</td>"
            $Html += "<td style=`"$CellStyle`">$($AgentJob.tests_passed_count)</td>"
            $Html += "<td style=`"$CellStyle`">$($AgentJob.tests_failed_count)</td>"
            $Html += "<td style=`"$CellStyle`">$($AgentJob.tests_skipped_count)</td>"
            $Html += "<td style=`"$CellStyle`">$($AgentJob.tests_timed_out_count)</td>"
            $Html += "<td style=`"$CellStyle`">$Time</td>"
            $Html += "</tr>"
        }
    }

    $Html += "</table><br>"
}

$Html | Out-File $OutputHtmlPath -Encoding utf8
Write-Host "##vso[task.uploadsummary]$(Resolve-Path $OutputHtmlPath)"

# ---------------------------------------------------------------------------------------------------------------------
# Email Variables
# ---------------------------------------------------------------------------------------------------------------------

if ($EmailTo)
{
    # Save comma-separated list of email addresses in the "emailTo" ADO pipeline variable.
    Write-Host "##vso[task.setVariable variable=emailTo;isOutput=true]$EmailTo"
    
    # Save email title in the "emailSubject" ADO pipeline variable.
    $OverallResult = $BuildResult
    $EmailSubject = "$($Run.Definition.Name) (${env:BUILD_REASON}:$ShortBranchName) - $($OverallResult)"
    Write-Host "##vso[task.setVariable variable=emailSubject;isOutput=true]$emailSubject"
}