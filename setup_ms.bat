@echo off

set HTTP_PROXY=http://192.168.8.3:3128
set HTTPS_PROXY=http://192.168.8.3:3128
set NO_PROXY=localhost,127.0.0.1,192.168.1.0/24,192.168.2.0/24,192.168.7.0/24

set CUDA=C:\_Dev\Repository\_CudaToolKit\v11.8
set CUDA_HOME=C:\_Dev\Repository\_CudaToolKit\v11.8
set CUDA_PATH=C:\_Dev\Repository\_CudaToolKit\v11.8

set PYTHON=C:\_Dev\Repository\Python\Python-3.10.11\python.exe
SET PATH=%PATH%;C:\UserData\Appz\VS2022\BuildTools\VC\Tools\MSVC\14.33.31629\bin\Hostx64\x64

set VENV_DIR=

IF NOT EXIST venv (
    set PYTHON=C:\_Dev\Repository\Python\Python-3.10.11\python.exe
    %PYTHON% -m venv venv
    call .\venv\Scripts\activate.bat
    python.exe -m pip install --upgrade pip
    rem install pytorch
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
    rem setup gaussian-splatting
    pip install -q plyfile
    cd /D %~dp0\repositories\gaussian-splatting\submodules\diff-gaussian-rasterization
    python setup.py install
    cd /D %~dp0\repositories\gaussian-splatting\submodules\simple-knn
    python setup.py install
    REM pip3 install -r requirements.txt
    pip3 install gradio
    
) ELSE (
    set PYTHON=.\venv\Scripts\python.exe
    echo venv folder already exists, skipping creation...
    call .\venv\Scripts\activate.bat
)

call .\venv\Scripts\activate.bat

REM pip install -r requirements_m.txt
cmd /k
REM python.exe -m pip install --upgrade pip
REM pip3 install -r requirements.txt
