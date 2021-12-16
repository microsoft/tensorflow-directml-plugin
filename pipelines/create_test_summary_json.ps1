# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Creates an aggregate of all test results.
.DESCRIPTION
Each tests runs on multiple environments, where an environment is defined by the agent (hardware),
build architecture, and build configuration. For example, a run may produce 4 builds 
(e.g. x64.release, x64.debug, x86.release, x86.debug) that are tested by 8 agents for up to
8*4 = 32 environments. If there are 2000 tests, then there may be up to 2000*32 = 64,000 results.
This script will summarize test results across all environments so that each test has a single state:
- passed: test case passes on all environments
- failed: test case fails on at least one environment
- skipped: test case skips on all environments (one failure/blocked promotes the test to failed/blocked)
#>
param
(
    [string]$TestArtifactsPath,
    [string]$OutputPath = "$TestArtifactsPath\test_summary.json"
)

# Sort test results.
$Summary = @{'groups'=@(); 'tests'=@{};}
$TestGroupMap = @{}
$AgentTestSummaryPaths = (Get-ChildItem "$TestArtifactsPath/*/*/summary.*.json").FullName
foreach ($AgentTestSummaryPath in $AgentTestSummaryPaths)
{
    Write-Host "Parsing $AgentTestSummaryPath"
    $AgentTestSummary = (Get-Content $AgentTestSummaryPath -Raw) | ConvertFrom-Json

    $GroupName = ($AgentTestSummaryPath | Split-Path -Leaf) -replace 'summary\.(\w+)\.json', '$1'
    $AgentName = $AgentTestSummaryPath | Split-Path -Parent | Split-Path -Parent | Split-Path -Leaf
    $BuildName = $AgentTestSummaryPath | Split-Path -Parent | Split-Path -Leaf

    if (!$TestGroupMap[$GroupName])
    {
        $TestGroupMap[$GroupName] = @{'Name'=$GroupName; 'Agents'=@(); 'Tests'=@{}}
        $Summary.groups += $TestGroupMap[$GroupName]
    }

    $TestGroupMap[$GroupName].Agents +=
    @{
        'name'=$AgentName;
        'build'=$BuildName;
        'time_seconds'=$AgentTestSummary.time_seconds;
        'counts'=
        @{
            'passed'=$AgentTestSummary.cases_passed; 
            'failed'=$AgentTestSummary.cases_failed; 
            'skipped'=$AgentTestSummary.cases_skipped; 
        }
    }

    foreach ($ResultType in ('passed', 'failed', 'skipped'))
    {
        foreach ($TestCase in $AgentTestSummary["cases_${ResultType}"])
        {
            # Full test name format: <group>.<test>::<class>.<method>
            # Example: plugin.profiler_test::ProfilerTest::testXPlaneKernelEvents
            if (!$Summary.tests[$TestCase.name])
            {
                $Summary.tests[$TestCase.name] = @{
                    'state'='?'
                    'failed'=@(); 
                    'passed'=@(); 
                    'skipped'=@();
                }
            }
    
            $Summary.tests[$TestCase.name][$TestResult.Result] += "$AgentName!$BuildName";
        }
    }
}

# Determine each test's state.
foreach ($TestSummary in $Summary.Tests.Values)
{
    $Total = $TestSummary.Fail.Count + $TestSummary.Pass.Count + $TestSummary.Blocked.Count + $TestSummary.Skipped.Count

    if ($TestSummary.Fail.Count -gt 0)
    {
        $TestSummary.State = 'Fail'
    }
    elseif ($TestSummary.Blocked.Count -gt 0)
    {
        $TestSummary.State = 'Blocked'
    }
    elseif ($TestSummary.Skipped.Count -eq $Total)
    {
        $TestSummary.State = 'Skipped'
    }
    else
    {
        $TestSummary.State = 'Pass'
    }
}

New-Item -ItemType File -Path $OutputPath -Force
$Summary | ConvertTo-Json -Depth 8 -Compress | Out-File $OutputPath -Encoding utf8