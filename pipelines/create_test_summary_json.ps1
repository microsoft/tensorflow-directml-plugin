# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Creates a summary of all test groups across all test environments.

.DESCRIPTION
Each tests runs on multiple environments, where an environment is defined by the agent (hardware),
build architecture, and build configuration. For example, a run may produce 4 builds 
(e.g. x64.release, x64.debug, x86.release, x86.debug) that are tested by 8 agents for up to
8*4 = 32 environments. If there are 2000 tests, then there may be up to 2000*32 = 64,000 results.
#>
param
(
    [string]$TestArtifactsPath,
    [string]$OutputPath = "$TestArtifactsPath\test_summary.json"
)

$Summary = @{}

$TestMatrix = (Get-Content "$TestArtifactsPath/matrix.json" -Raw) | ConvertFrom-Json

foreach ($Job in $TestMatrix)
{
    foreach ($TestConfig in $Job.agentTestConfigs)
    {
        $Build, $TensorflowPackage, $Group = $TestConfig.split(':')
        $ResultsPath = "$TestArtifactsPath/$($Job.agentName)/$Build/$TensorflowPackage/summary.$Group.json"

        if (!$Summary.ContainsKey($Group))
        {
            $Summary[$Group] = @()
        }

        $SummaryEntry = @{
            "agent" = $Job.agentName;
            "build" = "${Build}-${TensorflowPackage}";
            "agentWasOnline" = $Job.agentStatus -eq 'online';
            "agentWasEnabled" = $Job.agentEnabled;
            "agentHasResults" = Test-Path $ResultsPath;
        }

        if (Test-Path $ResultsPath)
        {
            $EnvSummary = (Get-Content $ResultsPath -Raw) | ConvertFrom-Json
            $SummaryEntry["tests_total_count"] = $EnvSummary.tests_total_count
            $SummaryEntry["tests_passed_count"] = $EnvSummary.tests_passed_count
            $SummaryEntry["tests_failed_count"] = $EnvSummary.tests_failed_count
            $SummaryEntry["tests_skipped_count"] = $EnvSummary.tests_skipped_count
            $SummaryEntry["tests_timed_out_count"] = $EnvSummary.tests_timed_out_count
            $SummaryEntry["start_timestamp_seconds"] = $EnvSummary.start_timestamp_seconds
            $SummaryEntry["end_timestamp_seconds"] = $EnvSummary.end_timestamp_seconds
            $SummaryEntry["duration_seconds"] = $EnvSummary.duration_seconds
        }
        
        $Summary[$Group] += $SummaryEntry
    }
}

New-Item -ItemType File -Path $OutputPath -Force
$Summary | ConvertTo-Json -Depth 8 -Compress | Out-File $OutputPath -Encoding utf8