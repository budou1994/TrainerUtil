@echo off
chcp 65001 > nul
title ImageProcessor

rem 检查Python环境
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python环境，请先安装Python 3.10或更高版本
    pause
    exit /b
)

rem 检查依赖
echo [信息] 检查依赖...
python -c "import flet" > nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安装依赖...
    pip install -q flet pillow opencv-python numpy torch torchvision insightface transformers onnxruntime
)

rem 检查models目录
if not exist "models" (
    echo [信息] 创建models目录...
    mkdir models
)

rem 检查logs目录
if not exist "logs" (
    echo [信息] 创建logs目录...
    mkdir logs
)

rem 启动程序
echo [信息] 启动程序...
python main.py

pause 