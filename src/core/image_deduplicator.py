import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
from src.utils.logger_config import setup_logger
import os

class ImageDeduplicator:
    def __init__(self, folder_path, progress_callback=None):
        self.folder_path = Path(folder_path)
        self.logger = logging.getLogger("image_deduplicator")
        self.progress_callback = progress_callback

    def remove_duplicates(self):
        """删除重复图片"""
        try:
            # 收集所有图片
            image_files = list(self.folder_path.rglob("*.jpg"))
            total_files = len(image_files)
            
            if self.progress_callback:
                self.progress_callback(0, total_files, "正在计算图片哈希...")
            
            # 计算哈希值
            hash_dict = {}
            for idx, img_path in enumerate(image_files, 1):
                try:
                    img_hash = self._calculate_hash(img_path)
                    if img_hash in hash_dict:
                        # 删除重复文件
                        os.remove(img_path)
                        self.logger.info(f"删除重复图片: {img_path.name}")
                        if self.progress_callback:
                            self.progress_callback(
                                idx, 
                                total_files, 
                                f"删除重复图片: {img_path.name}"
                            )
                    else:
                        hash_dict[img_hash] = img_path
                        if self.progress_callback:
                            self.progress_callback(
                                idx, 
                                total_files, 
                                f"处理图片: {img_path.name}"
                            )
                except Exception as e:
                    self.logger.error(f"处理图片失败 {img_path}: {str(e)}")
                    
            if self.progress_callback:
                removed_count = total_files - len(hash_dict)
                self.progress_callback(
                    total_files,
                    total_files,
                    f"重复图片处理完成 - 共处理: {total_files}, 删除重复: {removed_count}"
                )
                    
        except Exception as e:
            self.logger.error(f"删除重复图片失败: {str(e)}")
            raise

    def _calculate_hash(self, image_path: Path) -> str:
        """计算图片的感知哈希值"""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 调整大小为8x8
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        
        # 计算平均值
        avg = resized.mean()
        
        # 计算哈希值
        hash_value = ''
        for i in range(8):
            for j in range(8):
                hash_value += '1' if resized[i, j] > avg else '0'
        
        return hash_value 