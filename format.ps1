function ShouldFormat($FileName)
{
    return $FileName -clike "*.h" `
        -or $FileName -clike "*.c" `
        -or $FileName -clike "*.cpp" `
        -or $FileName -clike "*.cc"
}

function FormatFiles($Root)
{
    if (Test-Path -Path $Root -PathType Leaf)
    {
        if (ShouldFormat($Root))
        {
            Write-Output "Formatting $($Root)"
            clang-format -i --style=file $File.Path
        }
        continue
    }

    $Folder = (New-Object -Com Scripting.FileSystemObject).GetFolder($Root)
    $Files = $Folder.Files | Where-Object { ShouldFormat($_.Name) }
    foreach ($File in $Files)
    {
        Write-Output "Formatting $($File.Path)"
        clang-format -i --style=file $File.Path
    }
    foreach ($SubFolder in $Folder.Subfolders)
    {
        $CurrentItem = Get-Item $SubFolder.Path -ErrorAction SilentlyContinue
        if ($CurrentItem -and !$CurrentItem.Attributes.ToString().Contains("ReparsePoint"))
        {
            FormatFiles($SubFolder.Path)
        }
    }
}

$ErrorActionPreference = "Stop"

foreach ($Item in Get-ChildItem $PSScriptRoot)
{
    if ($Item.Name -ne "build" -and $Item.Name -ne "third_party")
    {
        FormatFiles($Item.FullName)
    }
}

black .
pylint build.py
pylint generate_op_defs_core.py
pylint test
pylint tfdml

Write-Output "Done!"