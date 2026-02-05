@echo off
REM VREyeSAM Setup Script for Windows
REM This script sets up the environment and downloads required files

echo ============================================================
echo VREyeSAM Windows Setup Script
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11 from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Creating virtual environment...
if exist vreyesam_env (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv vreyesam_env
    echo Done!
)
echo.

echo [2/6] Activating virtual environment...
call vreyesam_env\Scripts\activate.bat
echo Done!
echo.

echo [3/6] Installing dependencies...
echo This may take a few minutes...
python -m pip install --upgrade pip
pip install streamlit
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0.0"
pip install opencv-python-headless pillow pandas scikit-learn matplotlib tqdm hydra-core
echo Done!
echo.

echo [4/6] Cloning SAM2 repository...
if exist segment-anything-2 (
    echo SAM2 repository already exists, skipping...
) else (
    git clone https://github.com/facebookresearch/segment-anything-2
    echo Done!
)
echo.

echo [5/6] Installing SAM2...
cd segment-anything-2
pip install -e .
cd ..
echo Done!
echo.

echo [6/6] Downloading model checkpoints...
if not exist segment-anything-2\checkpoints mkdir segment-anything-2\checkpoints

REM Download SAM2 base checkpoint
if exist segment-anything-2\checkpoints\sam2_hiera_small.pt (
    echo SAM2 checkpoint already exists, skipping...
) else (
    echo Downloading SAM2 checkpoint (this may take a few minutes)...
    powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt' -OutFile 'segment-anything-2\checkpoints\sam2_hiera_small.pt'"
    echo Done!
)

REM Download VREyeSAM weights
if exist segment-anything-2\checkpoints\VREyeSAM_uncertainity_best.torch (
    echo VREyeSAM weights already exist, skipping...
) else (
    echo Downloading VREyeSAM weights...
    pip install huggingface-hub
    huggingface-cli download devnagaich/VREyeSAM VREyeSAM_uncertainity_best.torch --local-dir segment-anything-2\checkpoints\
    echo Done!
)
echo.

echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To run the app:
echo   1. Activate the environment: vreyesam_env\Scripts\activate.bat
echo   2. Run: streamlit run app.py
echo.
echo Press any key to exit...
pause >nul