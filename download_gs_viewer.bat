:: Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
::
:: NVIDIA CORPORATION and its licensors retain all intellectual property
:: and proprietary rights in and to this software, related documentation
:: and any modifications thereto.  Any use, reproduction, disclosure or
:: distribution of this software and related documentation without an express
:: license agreement from NVIDIA CORPORATION is strictly prohibited.

@echo off

set cwd=%cd%
cd /D %~dp0

echo Downloading gaussian-splatting-viewer...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip', 'viewers.zip')"

echo Unzipping...
powershell Expand-Archive viewers.zip -DestinationPath .\external\gs-viewer -Force

echo Cleaning up...
if exist viewers.zip del /f /q viewers.zip
exit /b
