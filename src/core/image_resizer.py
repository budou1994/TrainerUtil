import cv2
import os
import logging
from pathlib import Path
import numpy as np
import shutil
import tempfile

logger = logging.getLogger("image_resizer")

class ImageResizer:
    """图片尺寸调整器"""
    
    def __init__(self, max_dimension=3000, progress_callback=None):
        """
        初始化
        @param max_dimension: 最大边长限制
        @param progress_callback: 进度回调函数，接收参数(current, total, message)
        """
        self.max_dimension = max_dimension
        self.progress_callback = progress_callback
            
    def process_image(self, image_path):
        """
        处理单个图片
        @param image_path: 图片路径
        @return: 成功返回True，失败返回False
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"图片不存在: {image_path}")
                return False
                
            # 读取图片
            img = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            if img is None:
                logger.error(f"无法读取图片: {image_path}")
                return False
            
            # 获取当前图片尺寸
            height, width = img.shape[:2]
            max_current_dimension = max(width, height)
            
            # 判断是否需要调整尺寸
            if max_current_dimension <= self.max_dimension:
                logger.info(f"跳过图片(尺寸已合适) {image_path.name}: {width}×{height}")
                return True
                
            # 计算缩放比例
            scale = self.max_dimension / max_current_dimension
            
            # 计算新尺寸（保持宽高比）
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 调整尺寸
            resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            logger.info(f"调整图片尺寸 {image_path.name}: {width}×{height} -> {new_width}×{new_height}")
            
            # 使用临时文件保存
            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    temp_path = tmp_file.name
                
                # 编码并保存到临时文件
                _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if buffer is None:
                    logger.error(f"图片编码失败: {image_path}")
                    return False
                    
                # 写入临时文件
                with open(temp_path, 'wb') as f:
                    f.write(buffer.tobytes())
                
                # 备份原文件
                backup_path = str(image_path) + ".bak"
                shutil.copy2(str(image_path), backup_path)
                
                try:
                    # 替换原文件
                    shutil.move(temp_path, str(image_path))
                    # 删除备份
                    os.remove(backup_path)
                    logger.info(f"已保存调整后的图片: {image_path.name}")
                    return True
                    
                except Exception as e:
                    # 恢复备份
                    if os.path.exists(backup_path):
                        shutil.move(backup_path, str(image_path))
                    logger.error(f"替换文件失败: {image_path}, 错误: {str(e)}")
                    return False
                    
            except Exception as e:
                logger.error(f"保存图片失败: {image_path}, 错误: {str(e)}")
                # 清理临时文件
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
                
        except Exception as e:
            logger.error(f"处理图片失败: {image_path}, 错误: {str(e)}", exc_info=True)
            return False
            
    def process_directory(self, input_dir):
        """
        处理整个目录
        @param input_dir: 输入目录
        @return: (成功数量, 失败数量)
        """
        success_count = 0
        fail_count = 0
        
        input_dir = Path(input_dir)
        if not input_dir.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return 0, 0
        
        # 首先收集所有需要处理的文件
        image_files = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        for image_path in input_dir.rglob("*"):
            if image_path.suffix.lower() in supported_formats:
                image_files.append(image_path)
        
        total_files = len(image_files)
        if total_files == 0:
            return 0, 0
            
        # 处理文件
        for index, image_path in enumerate(image_files, 1):
            if self.progress_callback:
                self.progress_callback(
                    index, 
                    total_files,
                    f"正在处理: {image_path.name} ({index}/{total_files})"
                )
                
            if self.process_image(image_path):
                success_count += 1
            else:
                fail_count += 1
        
        # 完成处理
        if self.progress_callback:
            self.progress_callback(
                total_files,
                total_files,
                f"处理完成 - 成功: {success_count}, 失败: {fail_count}"
            )
            
        return success_count, fail_count 