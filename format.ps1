function FormatFiles($Root)
{
    $Folder = (New-Object -Com Scripting.FileSystemObject).GetFolder($Root)
    $Files = $Folder.Files | Where-Object {
        $_.Name -clike "*.h" `
        -or $_.Name -clike "*.c" `
        -or $_.Name -clike "*.cpp" `
        -or $_.Name -clike "*.cc"
    }
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
            FormatFiles($SubFolder.Path);
        }
    }
}

$ErrorActionPreference = "Stop"
FormatFiles($PSScriptRoot)

Write-Output "Linting and formatting bazel files"
bazel run //:buildifier

Write-Output "Done!"