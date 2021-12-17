# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Converts an aggregate of all JSON test summaries into the xUnit format for VSTS.

.DESCRIPTION
This script then reads all the JSON results, builds an aggregate summary, and outputs an 
xUnit-formatted file that VSTS can use to display test results in the browser.

This script uses the following rules to aggregate the results:
- Test groups are reported as test assemblies.
- Test modules are reported as test collections.
- A test result is 'pass' if it passes on at least one agent, and has not failed on any agents.
- A test result is 'fail' if at least one agent fails the test.
- A test result is 'skipped' if all agents report either 'skipped' or 'blocked' in TAEF.
- A failing test's "errors" element contains the errors for all failing agents.
- The run date & time are the earliest run date and time from all agents.
- Test times are the median time from all agent test times.
- Assembly (test group) times are reported as the median of all agent group times.

xUnit schema: https://xunit.github.io/docs/format-xml-v2.html
#>
param
(
    [string]$TestArtifactsPath,
    [string]$OutputPath = "$TestArtifactsPath\test_summary.xml"
)

$AllSummaryFiles = (Get-ChildItem "$TestArtifactsPath\*\*\summary.*.json")
$Groups = $AllSummaryFiles.Name -replace 'summary\.(\w+)\.json', '$1' | Select-Object -Unique

$XmlMemoryStream = [System.IO.MemoryStream]::new()
$XmlWriterSettings = [System.Xml.XmlWriterSettings]::new()
$XmlWriterSettings.Indent = $true
$XmlWriterSettings.Encoding = [System.Text.UTF8Encoding]::new($false)
$XmlWriter = [System.Xml.XmlWriter]::Create($XmlMemoryStream, $XmlWriterSettings)
$XmlWriter.WriteStartDocument()
$XmlWriter.WriteStartElement("assemblies")

foreach ($Group in $Groups)
{
    $TestResults = @{}
    $GroupSummaryFiles = $AllSummaryFiles | Where-Object Name -eq "summary.${Group}.json"
    foreach ($AgentSummaryFile in $GroupSummaryFiles)
    {
        Write-Host "Parsing $AgentSummaryFile"

        $Summary = Get-Content $AgentSummaryFile.FullName -Raw | ConvertFrom-Json
        # $BuildName = $AgentSummaryFile.FullName | Split-Path -Parent | Split-Path -Leaf
        # $AgentName = $AgentSummaryFile.FullName | Split-Path -Parent | Split-Path -Parent | Split-Path -Leaf

        foreach ($Test in $Summary.tests_skipped)
        {
            $State = $TestResults[$Test]
            if (!$State -or (($State -ne 'passed') -and ($State -ne 'failed'))) 
            { 
                $TestResults[$Test] = 'skipped' 
            }
        }

        foreach ($Test in $Summary.tests_passed)
        {
            $State = $TestResults[$Test]
            if (!$State -or ($State -ne 'failed')) 
            { 
                $TestResults[$Test] = 'passed' 
            }
        }

        foreach ($Test in $Summary.tests_failed)
        {
            $TestResults[$Test] = 'failed' 
        }
    }

    $Total = $TestResults.Count
    $Passed = 0
    $Failed = 0
    $Skipped = 0
    foreach ($Test in $TestResults.Keys)
    {
        $State = $TestResults[$Test]
        if ($State -eq 'passed') { $Passed += 1}
        if ($State -eq 'skipped') { $Skipped += 1}
        if ($State -eq 'failed') 
        { 
            $Failed += 1
        }
    }

    $XmlWriter.WriteStartElement("assembly")
    $XmlWriter.WriteAttributeString("name", "autopilot::$($this.Name)")
    $XmlWriter.WriteAttributeString("test-framework", "abseil")
    $XmlWriter.WriteAttributeString("total", $Total)
    $XmlWriter.WriteAttributeString("passed", $Passed)
    $XmlWriter.WriteAttributeString("failed", $Failed)
    $XmlWriter.WriteAttributeString("skipped", $Skipped)
    $XmlWriter.WriteAttributeString("errors", 0)

    $XmlWriter.WriteStartElement("collection")
    $XmlWriter.WriteAttributeString("name", $this.Name)
    $XmlWriter.WriteAttributeString("total", $Total)
    $XmlWriter.WriteAttributeString("passed", $Passed)
    $XmlWriter.WriteAttributeString("failed", $Failed)
    $XmlWriter.WriteAttributeString("skipped", $Skipped)

    foreach ($Test in $TestResults.Keys)
    {
        $State = $TestResults[$Test]

        $XmlWriter.WriteStartElement("test")
        $XmlWriter.WriteAttributeString("name", $Test)
        $XmlWriter.WriteAttributeString("method", $Test)
        # $XmlWriter.WriteAttributeString("time", $this.MedianTime)
        $XmlWriter.WriteAttributeString("result", $State)

        # if (($this.State -ne 'Pass') -and ($this.State -ne 'Skipped'))
        # {
        #     $XmlWriter.WriteStartElement("failure")
        #     $XmlWriter.WriteStartElement("message")
        #     $XmlWriter.WriteCData(($this.Errors -join "`n`n"))
        #     $XmlWriter.WriteEndElement() # message
        #     $XmlWriter.WriteEndElement() # failure
        # }

        $XmlWriter.WriteEndElement() # test
    }

    $XmlWriter.WriteEndElement() # collection
    $XmlWriter.WriteEndElement() # assembly
}

$XmlWriter.WriteEndElement() # assemblies
$XmlWriter.Flush()
$XmlWriter.Close()

Write-Host 'Saving XML file...'
New-Item -ItemType File -Path $OutputPath -Force
[System.Text.Encoding]::UTF8.GetString($XmlMemoryStream.ToArray()) | Out-File $OutputPath -Encoding utf8