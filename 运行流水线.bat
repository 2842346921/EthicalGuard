@echo off
rem ============================================================
rem  EthicalGuard 完整流水线（③数据准备 → ④协商 → ⑤韧性 → ⑥评估）
rem  双击运行本文件即可；也可在命令行执行同款 python 命令。
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ============ STEP 3/6  Data prepare (01_prepare) ============
python scripts\01_prepare.py --config configs\config.yaml --limit 50
if errorlevel 1 goto :err

echo.
echo ============ STEP 4/6  MANE negotiation (02_run_mane) ============
python scripts\02_run_mane.py --config configs\config.yaml --input data_cache\scenarios.jsonl --limit 10
if errorlevel 1 goto :err

echo.
echo ============ STEP 5/6  Resilience stress (03_stress) ============
python scripts\03_stress.py --config configs\config.yaml --input data_cache\scenarios.jsonl --limit 10
if errorlevel 1 goto :err

echo.
echo ============ STEP 6/6  Evaluation report (04_eval) ============
python scripts\04_eval.py --config configs\config.yaml
if errorlevel 1 goto :err

echo.
echo ============================================================
echo  ALL DONE! Results are in:
echo    data_cache\scenarios.jsonl   (scenarios)
echo    runs\mane_results.jsonl      (negotiation + KKT + resource)
echo    runs\resilience.jsonl        (resilience reports)
echo ============================================================
pause
exit /b 0

:err
echo.
echo  ************************************************************
echo  *  ERROR! Please copy the error message above to the user.  *
echo  ************************************************************
pause
exit /b 1
