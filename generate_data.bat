@echo off
echo ====================================================================
echo KP Developmental Landscape Datasets - Generator & Baseline Model
echo ====================================================================
echo.
echo Step 1: Generating full 300,000 synthetic records...
python generate_data.py
echo.
echo Step 2: Running baseline regression model training...
python train_baseline.py
echo.
echo Done! Please review the console output.
pause
