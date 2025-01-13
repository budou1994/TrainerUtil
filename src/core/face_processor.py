import cv2
import numpy as np
from pathlib import Path
import logging
import insightface
from insightface.app import FaceAnalysis
import time
from src.utils.logger_config import setup_logger

class FaceProcessor:
    def __init__(self, target_size=1024, face_only=False, face_features=False, progress_callback=None):
        self.target_size = target_size
        self.face_only = face_only
        self.face_features = face_features
        self.logger = logging.getLogger("face_processor")
        self.progress_callback = progress_callback
        self._init_model()

    def _init_model(self):
        # 初始化 InsightFace
        self.app = FaceAnalysis(
            name='buffalo_l',
            root='models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 基础检测参数
        self.min_face_score = 0.4  # 降低最小检测置信度
        self.min_face_size = 40    # 降低最小人脸尺寸
        
        # 不同场景的裁剪参数
        self.crop_params = {
            'head_shoulder': {
                'scale': 3.0,        # 更大的裁剪范围，包含更多上半身
                'shift_up': 0.3,     # 向上偏移以包含更多头部区域
                'shift_down': 0.7,   # 向下偏移以包含更多肩部
                'width_scale': 1.4   # 横向扩展以包含更多背景
            },
            'face_only': {
                'scale': 2.2,        # 适中的裁剪范围，主要包含面部
                'shift_up': 0.2,     # 稍微向上以包含发际线
                'shift_down': 0.2,   # 稍微向下以包含下巴
                'width_scale': 1.2   # 适当横向扩展
            },
            'features': {
                'scale': 1.5,        # 较小的裁剪范围，集中于五官
                'shift_up': 0.1,     # 最小的上下偏移
                'shift_down': 0.1,
                'width_scale': 1.1   # 最小的横向扩展
            }
        }

    def _get_face_box(self, face, img_size):
        """根据人脸关键点计算裁剪框"""
        # 获取人脸关键点的边界
        landmarks = face.kps
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        # 选择裁剪参数
        if self.face_features:
            params = self.crop_params['features']
        elif self.face_only:
            params = self.crop_params['face_only']
        else:
            params = self.crop_params['head_shoulder']
        
        # 计算人脸中心和基础大小
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        face_size = max(x_max - x_min, y_max - y_min)
        
        # 应用缩放和偏移
        crop_size = int(face_size * params['scale'])
        shift_up = int(crop_size * params['shift_up'])
        shift_down = int(crop_size * params['shift_down'])
        width_scale = params['width_scale']
        
        # 计算裁剪区域
        y1 = max(0, int(center_y - crop_size/2 - shift_up))
        y2 = min(img_size[0], int(center_y + crop_size/2 + shift_down))
        
        # 计算横向扩展
        width = int(crop_size * width_scale)
        x1 = max(0, int(center_x - width/2))
        x2 = min(img_size[1], int(center_x + width/2))
        
        # 调整以保持比例
        current_height = y2 - y1
        current_width = x2 - x1
        target_size = max(current_height, current_width)
        
        # 确保裁剪框是正方形
        if current_width < target_size:
            diff = target_size - current_width
            x1 = max(0, x1 - diff//2)
            x2 = min(img_size[1], x1 + target_size)
        if current_height < target_size:
            diff = target_size - current_height
            y1 = max(0, y1 - diff//2)
            y2 = min(img_size[0], y1 + target_size)
            
        return int(x1), int(y1), int(x2), int(y2)

    def _detect_faces(self, image_path):
        """检测图片中的人脸"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
                
            # 检测人脸
            faces = self.app.get(image)
            
            # 按置信度排序
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
            
            # 过滤低置信度和小尺寸的人脸
            valid_faces = []
            for face in faces:
                if face.det_score < self.min_face_score:
                    continue
                    
                bbox = face.bbox.astype(int)
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                
                if face_width < self.min_face_size or face_height < self.min_face_size:
                    continue
                    
                valid_faces.append(face)
            
            return valid_faces
            
        except Exception as e:
            self.logger.error(f"人脸检测失败: {str(e)}")
            raise

    def _save_face(self, image_path, face, face_idx):
        """保存检测到的人脸"""
        try:
            # 读取原图
            image = cv2.imread(str(image_path))
            
            # 获取裁剪区域
            x1, y1, x2, y2 = self._get_face_box(face, image.shape[:2])
            
            # 裁剪图片
            cropped = image[y1:y2, x1:x2]
            
            # 调整大小
            resized = cv2.resize(cropped, (self.target_size, self.target_size),
                               interpolation=cv2.INTER_LANCZOS4)
            
            # 创建输出目录
            output_dir = Path(image_path).parent / "faces"
            output_dir.mkdir(exist_ok=True)
            
            # 保存结果
            output_path = output_dir / f"{Path(image_path).stem}_face_{face_idx+1}.jpg"
            cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"保存人脸失败: {str(e)}")
            raise

    def process_folder(self, folder_path):
        """处理文件夹中的所有图片"""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"文件夹不存在: {folder_path}")

        # 收集所有图片
        image_files = list(folder.rglob("*.jpg"))
        total_files = len(image_files)
        
        if self.progress_callback:
            self.progress_callback(0, total_files, "开始处理人脸...")
        
        success_count = 0
        for idx, image_path in enumerate(image_files, 1):
            try:
                faces = self._detect_faces(str(image_path))
                if faces:
                    for face_idx, face in enumerate(faces):
                        output_path = self._save_face(image_path, face, face_idx)
                        self.logger.info(f"保存人脸: {output_path}")
                        success_count += 1
                else:
                    self.logger.info(f"未检测到人脸: {image_path.name}")
                        
                if self.progress_callback:
                    self.progress_callback(
                        idx,
                        total_files,
                        f"处理图片: {image_path.name} - 检测到 {len(faces)} 个人脸"
                    )
                    
            except Exception as e:
                self.logger.error(f"处理图片失败 {image_path}: {str(e)}")
                
        if self.progress_callback:
            self.progress_callback(
                total_files,
                total_files,
                f"人脸处理完成 - 共处理: {total_files}张图片, 提取: {success_count}个人脸"
            ) 