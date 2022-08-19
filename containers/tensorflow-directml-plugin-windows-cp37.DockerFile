FROM mcr.microsoft.com/windows:ltsc2019
SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Install python
RUN $Url = 'https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe'
RUN $DownloadPath = "$env:TEMP/python-3.7.9-amd64.exe"
RUN (New-Object System.Net.WebClient).DownloadFile($Url, $DownloadPath)
RUN $env:TEMP/python-3.7.9-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
RUN Start-Process $DownloadPath -ArgumentList '/quiet', 'InstallAllUsers=1', 'PrependPath=1', 'Include_test=0' -Wait