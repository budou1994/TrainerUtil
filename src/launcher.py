import os
import sys
from pathlib import Path

def init_environment():
    """初始化运行环境"""
    # 设置程序运行目录
    if getattr(sys, 'frozen', False):
        # 如果是打包后的exe
        app_dir = Path(sys._MEIPASS)
    else:
        # 如果是源码运行
        app_dir = Path(__file__).parent.parent
    
    os.chdir(str(app_dir))
    
    # 创建必要的目录
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

def main():
    """程序入口"""
    init_environment()
    
    # 导入并运行主程序
    from src.gui.main_window import ImageProcessorGUI
    import flet as ft
    
    ft.app(target=lambda page: ImageProcessorGUI(page).initialize())

if __name__ == "__main__":
    main() 