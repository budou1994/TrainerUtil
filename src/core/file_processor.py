import os
import shutil
from pathlib import Path
from src.utils.logger_config import setup_logger

class FileProcessor:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.logger = setup_logger("file_processor")
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

    def collect_files(self):
        """收集所有图片文件到根目录"""
        try:
            # 遍历所有子目录
            for root, _, files in os.walk(self.folder_path):
                root_path = Path(root)
                if root_path == self.folder_path:
                    continue
                
                # 移动图片文件
                for file in files:
                    if Path(file).suffix.lower() in self.supported_formats:
                        src = root_path / file
                        dst = self.folder_path / file
                        # 如果目标文件已存在，添加序号
                        if dst.exists():
                            base = dst.stem
                            suffix = dst.suffix
                            counter = 1
                            while dst.exists():
                                dst = self.folder_path / f"{base}_{counter}{suffix}"
                                counter += 1
                        shutil.move(str(src), str(dst))
                        self.logger.info(f"已移动: {src} -> {dst}")

            # 清理空文件夹
            for root, dirs, files in os.walk(self.folder_path, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        self.logger.info(f"已删除空文件夹: {dir_path}")

        except Exception as e:
            self.logger.error(f"文件收集失败: {str(e)}")
            raise 