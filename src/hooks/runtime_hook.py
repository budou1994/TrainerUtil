import os 
import sys 
from pathlib import Path 
 
def runtime_init(): 
    if getattr(sys, 'frozen', False): 
        app_dir = Path(sys._MEIPASS) 
        os.environ['PATH'] = f"{app_dir};{os.environ['PATH']}" 
        os.chdir(app_dir) 
        for dir_name in ['logs', 'models', 'output']: 
            Path(dir_name).mkdir(exist_ok=True) 
 
runtime_init() 
