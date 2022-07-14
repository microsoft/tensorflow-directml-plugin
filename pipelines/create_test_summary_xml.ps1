# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Converts an aggregate of all JSON test group summaries into the xUnit format for ADO.

.DESCRIPTION
This script then reads all the JSON results, builds an aggregate summary, and outputs an 
xUnit-formatted file that VSTS can use to display test results in the browser.

xUnit schema: https://xunit.net/docs/format-xml-v2
#>
param
(
    [string]$TestArtifactsPath,
    [string]$OutputPath = "$TestArtifactsPath\test_summary.xml",
    [int]$MaxTestCasesReportedPerTestFailure = 1000,
    [int]$MaxLogLinesReportedPerTestFailure = 50
)

$AllSummaryFiles = (Get-ChildItem "$TestArtifactsPath\summary.*.json" -Recurse)
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
    $TestFailureMessages = @{}
    $FirstStartTime = -1
    $LastEndTime = -1
    
    $GroupSummaryFiles = $AllSummaryFiles | Where-Object Name -eq "summary.${Group}.json"
    foreach ($AgentSummaryFile in $GroupSummaryFiles)
    {
        Write-Host "Parsing $AgentSummaryFile"

        $AgentSummary = Get-Content $AgentSummaryFile.FullName -Raw | ConvertFrom-Json

        if (($FirstStartTime -eq -1) -or ($AgentSummary.start_timestamp_seconds -lt $FirstStartTime))
        {
            $FirstStartTime = $AgentSummary.start_timestamp_seconds
        }
        if (($LastEndTime -eq -1) -or ($AgentSummary.end_timestamp_seconds -gt $LastEndTime))
        {
            $LastEndTime = $AgentSummary.end_timestamp_seconds
        }

        $TensorflowPackageName = $AgentSummaryFile.FullName | Split-Path -Parent | Split-Path -Leaf
        $BuildName = $AgentSummaryFile.FullName | Split-Path -Parent | Split-Path -Parent | Split-Path -Leaf
        $AgentName = $AgentSummaryFile.FullName | Split-Path -Parent | Split-Path -Parent | Split-Path -Parent Split-Path -Leaf

        foreach ($Test in $AgentSummary.tests)
        {
            # The reported test result is a combination of the results on all agents.
            # - All agents 'skipped' -> SKIP
            # - At least one agent 'passed' and no agents 'failed' or 'timed_out' -> PASS
            # - At least one agent 'failed' or 'timed_out' -> FAIL

            $CurrentTestState = $TestResults[$Test.name]
            if ($Test.result -eq 'skipped')
            {
                # A test may only be 'skipped' if all agents skip it.
                if (!$CurrentTestState)
                { 
                    $TestResults[$Test.name] = 'Skip' 
                }
            }
            elseif ($Test.result -eq 'passed')
            {
                # A test may be promoted from 'skipped' to 'passed', but any failure takes priority.
                if ($CurrentTestState -ne 'Fail')
                { 
                    $TestResults[$Test.name] = 'Pass' 
                }
            }
            elseif ($Test.result -eq 'timed_out')
            {
                # A timeout on any agent fails the test.
                $TestResults[$Test.name] = 'Fail'
                if (!$TestFailureMessages[$Test.name]) 
                { 
                    $TestFailureMessages[$Test.name] = [Collections.ArrayList]::new() 
                }
                $TestFailureMessages[$Test.name].Add("# $AgentName ($BuildName, $TensorflowPackageName): Timed Out") | Out-Null
            }
            elseif ($Test.result -eq 'failed')
            {
                # A failure on any agent fails the test.
                $TestResults[$Test.name] = 'Fail'
                if (!$TestFailureMessages[$Test.name]) 
                { 
                    $TestFailureMessages[$Test.name] = [Collections.ArrayList]::new() 
                }

                $CaseResultsFile = "$TestArtifactsPath/$AgentName/$BuildName/$TensorflowPackageName/test.$($Test.name).xml"
                $LogResultsFile = "$TestArtifactsPath/$AgentName/$BuildName/$TensorflowPackageName/log.$($Test.name).txt"
    
                if ($Test.cases_failed -and (Test-Path $CaseResultsFile))
                {
                    # Prefer to list failure messages from the Abseil test log.
                    $TestFailureMessages[$Test.name].Add("## $AgentName ($BuildName, $TensorflowPackageName): Cases Failed") | Out-Null
        
                    [xml]$TestResultsXml = Get-Content $CaseResultsFile
    
                    $RelatedCasesToReport = $Test.cases_failed | Select-Object -First $MaxTestCasesReportedPerTestFailure
                    foreach ($Case in $RelatedCasesToReport)
                    {
                        $TestFailureMessages[$Test.name].Add("<b>$Case</b>") | Out-Null
        
                        if ($Case -match ".*::(\w+).(.*)")
                        {
                            $TestClass = $Matches[1]
                            $TestMethod = $Matches[2]
                            $ClassResults = $TestResultsXml.testsuites.testsuite | ? name -eq $TestClass
                            $CaseResults = $ClassResults.testcase | ? name -eq $TestMethod
                            $ErrorMessage = $CaseResults.error.message
                            if ($ErrorMessage)
                            {
                                $TestFailureMessages[$Test.name].Add($ErrorMessage) | Out-Null
                            }
                            $FailureMessage = $CaseResults.failure.message
                            if ($FailureMessage)
                            {
                                $TestFailureMessages[$Test.name].Add($FailureMessage) | Out-Null
                            }
                        }
                    }
    
                    if ($Test.cases_failed.Count -gt $RelatedCasesToReport.Count)
                    {
                        $AdditionalCount = $Test.cases_failed.Count - $RelatedCasesToReport.Count
                        $TestFailureMessages[$Test.name].Add("... (results truncated. See '$AgentName/$BuildName/$TensorflowPackageName/test.$Test.xml' for $AdditionalCount additional failures.)") | Out-Null
                    }
                }
                elseif (Test-Path $LogResultsFile)
                {
                    # The Abseil test log may not exist if the test aborted. Use the log instead (max 50 lines).
                    $TestFailureMessages[$Test.name].Add("## $AgentName ($BuildName): Aborted") | Out-Null
    
                    $Log = Get-Content $LogResultsFile
                    $Lines = $Log | Select-Object -First $MaxLogLinesReportedPerTestFailure
                    $TestFailureMessages[$Test.name].Add('```') | Out-Null
                    foreach ($Line in $Lines)
                    {
                        $TestFailureMessages[$Test.name].Add($Line) | Out-Null
                    }
                    if ($Log.Count -gt $MaxLogLinesReportedPerTestFailure)
                    {
                        $TestFailureMessages[$Test.name].Add("... (results truncated. See '$AgentName/$BuildName/$TensorflowPackageName/log.$Test.txt' for full log.)") | Out-Null
                    }
                    $TestFailureMessages[$Test.name].Add('```') | Out-Null
                }
                else
                {
                    $TestFailureMessages[$Test.name].Add("## $AgentName ($BuildName, $TensorflowPackageName): Unknown Errors") | Out-Null
                }
            }
        }
    }

    $Total = $TestResults.Count
    $Passed = 0
    $Failed = 0
    $Skipped = 0
    foreach ($Test in $TestResults.Keys)
    {
        $State = $TestResults[$Test]
        if ($State -eq 'Pass') { $Passed += 1}
        if ($State -eq 'Skip') { $Skipped += 1}
        if ($State -eq 'Fail') 
        { 
            $Failed += 1
        }
    }

    $StartDateTime = (Get-Date -Date "01/01/1970").AddSeconds($FirstStartTime)
    $EndDateTime = (Get-Date -Date "01/01/1970").AddSeconds($LastEndTime)
    $RunTime = $EndDateTime - $StartDateTime

    $XmlWriter.WriteStartElement("assembly")
    $XmlWriter.WriteAttributeString("name", $Group)
    $XmlWriter.WriteAttributeString("test-framework", "abseil")
    $XmlWriter.WriteAttributeString("run-date", $StartDateTime.ToString('yyyy-MM-dd'))
    $XmlWriter.WriteAttributeString("run-time", $RunTime)
    $XmlWriter.WriteAttributeString("total", $Total)
    $XmlWriter.WriteAttributeString("passed", $Passed)
    $XmlWriter.WriteAttributeString("failed", $Failed)
    $XmlWriter.WriteAttributeString("skipped", $Skipped)
    $XmlWriter.WriteAttributeString("errors", 0)

    $XmlWriter.WriteStartElement("collection")
    $XmlWriter.WriteAttributeString("name", $Group)
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
        $XmlWriter.WriteAttributeString("result", $State)

        $FailureMessage = $TestFailureMessages[$Test]
        if ($FailureMessage)
        {
            $XmlWriter.WriteStartElement("failure")
            $XmlWriter.WriteStartElement("message")
            $XmlWriter.WriteCData(($FailureMessage -join "`n"))
            $XmlWriter.WriteEndElement() # message
            $XmlWriter.WriteEndElement() # failure
        }

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