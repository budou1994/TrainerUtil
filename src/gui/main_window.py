import flet as ft
import logging
from pathlib import Path
from tkinter import filedialog
from src.core.image_captioner import ImageCaptioner
from src.core.image_deduplicator import ImageDeduplicator
from src.core.image_resizer import ImageResizer
from src.core.file_processor import FileProcessor
from src.core.face_processor import FaceProcessor
import threading
import torch
from src.utils.logger_config import setup_logger

class ImageProcessorGUI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "图片批处理工具"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.window_width = 1000
        self.page.window_height = 800
        self.page.padding = 0  # 移除页面边距
        self.page.scroll = "auto"  # 启用页面滚动
        self.page.fonts = {
            "楷体": "KaiTi"
        }
        
        # 初始化日志
        self.logger = setup_logger("gui")
        
        # 初始化变量
        self.initialize_variables()

    def initialize_variables(self):
        """初始化变量"""
        # 文件夹路径
        self.folder_path = ""
        
        # 任务选项
        self.collect_value = False
        self.deduplicate_value = False
        self.resize_value = False
        self.face_value = False
        self.face_only_value = False
        self.face_features_value = False
        self.caption_brief_value = False
        self.caption_normal_value = False
        self.caption_detailed_value = False

        # 存储关键控件的引用
        self.folder_text = None
        self.max_size_input = None
        self.face_size_dropdown = None
        self.progress_bar = None
        self.progress_text = None
        self.task_progress_text = None
        self.start_button = None
        self.task_containers = []

    def initialize(self):
        """初始化界面"""
        # 创建主容器
        self.main_container = ft.Container(
            content=ft.Column(
                controls=[
                    # 内容区域
                    ft.Container(
                        content=ft.Column(
                            controls=[
                                self.create_folder_section(),
                                self.create_task_section(),
                                self.create_progress_section(),
                            ],
                            spacing=20
                        ),
                        padding=ft.padding.all(30),
                        expand=True
                    ),
                    # 底部固定按钮区域
                    ft.Container(
                        content=self.create_control_section(),
                        padding=ft.padding.only(left=30, right=30, bottom=30),
                        bgcolor=ft.colors.WHITE,
                    )
                ],
                spacing=0,
                scroll="auto"  # 启用内容区域滚动
            ),
            expand=True
        )
        
        # 添加到页面
        self.page.add(self.main_container)

    def create_folder_section(self):
        """创建文件夹选择部分"""
        # 创建文件夹路径文本
        self.folder_text = ft.Text(
            value="请选择图片文件夹",
            size=16,
            font_family="楷体",
            expand=True
        )
        
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("文件夹选择", 
                           size=20, 
                           weight=ft.FontWeight.BOLD,
                           font_family="楷体"),
                    ft.Row([
                        self.folder_text,
                        ft.ElevatedButton(
                            text="选择文件夹",
                            on_click=self.select_folder,
                            style=ft.ButtonStyle(
                                padding=ft.padding.all(15)
                            )
                        )
                    ])
                ]),
                padding=20
            )
        )

    def create_task_section(self):
        """创建任务选择部分"""
        # 创建尺寸输入框
        self.max_size_input = ft.TextField(
            value="3000",
            width=150,
            text_align=ft.TextAlign.RIGHT,
            disabled=True
        )
        
        # 创建人脸尺寸下拉框
        self.face_size_dropdown = ft.Dropdown(
            width=150,
            options=[
                ft.dropdown.Option("1024×1024"),
                ft.dropdown.Option("512×512")
            ],
            disabled=True
        )
        
        # 创建基础处理容器
        basic_container = ft.Container(
            content=ft.Column([
                ft.Text("基础处理", 
                       size=16,
                       weight=ft.FontWeight.BOLD,
                       font_family="楷体"),
                ft.Checkbox(
                    label="收集文件（将所有图片移动到根目录并清理空文件夹）",
                    value=self.collect_value,
                    on_change=self.on_collect_changed
                ),
                ft.Checkbox(
                    label="删除重复图片",
                    value=self.deduplicate_value,
                    on_change=self.on_deduplicate_changed
                )
            ]),
            padding=10
        )
        
        # 创建尺寸调整容器
        resize_container = ft.Container(
            content=ft.Column([
                ft.Text("尺寸调整", 
                       size=16,
                       weight=ft.FontWeight.BOLD,
                       font_family="楷体"),
                ft.Checkbox(
                    label="调整图片尺寸",
                    value=self.resize_value,
                    on_change=self.on_resize_changed
                ),
                ft.Row([
                    ft.Text("最大尺寸:", size=14),
                    self.max_size_input,
                    ft.Text("像素", size=14),
                    ft.Text(
                        "（可手动输入或使用上下箭头调整，默认3000像素）",
                        size=12,
                        color=ft.colors.GREY_700
                    )
                ], alignment=ft.MainAxisAlignment.START)
            ]),
            padding=10
        )
        
        # 创建人脸处理容器
        face_container = ft.Container(
            content=ft.Column([
                ft.Text("人脸处理", 
                       size=16,
                       weight=ft.FontWeight.BOLD,
                       font_family="楷体"),
                ft.Checkbox(
                    label="人脸检测和裁剪（头肩部）",
                    value=self.face_value,
                    on_change=self.on_face_changed
                ),
                ft.Checkbox(
                    label="人脸检测和裁剪（仅人脸）",
                    value=self.face_only_value,
                    on_change=self.on_face_only_changed
                ),
                ft.Checkbox(
                    label="人脸检测和裁剪（仅五官）",
                    value=self.face_features_value,
                    on_change=self.on_face_features_changed
                ),
                ft.Row([
                    ft.Text("目标尺寸:", size=14),
                    self.face_size_dropdown
                ], alignment=ft.MainAxisAlignment.START)
            ]),
            padding=10
        )
        
        # 创建图片描述容器
        caption_container = ft.Container(
            content=ft.Column([
                ft.Text("图片描述", 
                       size=16,
                       weight=ft.FontWeight.BOLD,
                       font_family="楷体"),
                ft.Checkbox(
                    label="生成简略图片描述（关键特征描述）",
                    value=self.caption_brief_value,
                    on_change=self.on_caption_brief_changed
                ),
                ft.Checkbox(
                    label="生成标准图片描述（基本场景和主体描述）",
                    value=self.caption_normal_value,
                    on_change=self.on_caption_normal_changed
                ),
                ft.Checkbox(
                    label="生成详细图片描述（完整场景和细节描述）",
                    value=self.caption_detailed_value,
                    on_change=self.on_caption_detailed_changed
                )
            ]),
            padding=10
        )
        
        # 存储任务容器引用
        self.task_containers = [
            basic_container,
            resize_container,
            face_container,
            caption_container
        ]
        
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("任务选择", 
                           size=20, 
                           weight=ft.FontWeight.BOLD,
                           font_family="楷体"),
                    basic_container,
                    resize_container,
                    face_container,
                    caption_container
                ]),
                padding=20
            )
        )

    def create_progress_section(self):
        """创建进度显示部分"""
        self.progress_bar = ft.ProgressBar(
            width=800,
            height=20,
            value=0
        )
        
        self.progress_text = ft.Text(
            value="准备就绪",
            size=14,
            color=ft.colors.GREY_700,
            text_align=ft.TextAlign.CENTER,
            font_family="楷体"
        )
        
        self.task_progress_text = ft.Text(
            value="",
            size=14,
            color=ft.colors.GREY_700,
            text_align=ft.TextAlign.CENTER,
            font_family="楷体"
        )
        
        # 添加日志显示区域
        self.log_display = ft.ListView(
            height=150,
            spacing=2,
            padding=10,
            auto_scroll=True
        )
        
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("处理进度", 
                           size=20,
                           weight=ft.FontWeight.BOLD,
                           font_family="楷体"),
                    self.progress_bar,
                    self.progress_text,
                    self.task_progress_text,
                    ft.Divider(),  # 添加分隔线
                    ft.Text("处理日志", 
                           size=16,
                           weight=ft.FontWeight.BOLD,
                           font_family="楷体"),
                    ft.Container(  # 日志容器
                        content=self.log_display,
                        border=ft.border.all(1, ft.colors.GREY_300),
                        border_radius=5,
                        padding=5
                    )
                ]),
                padding=20
            )
        )

    def add_log(self, message, level="INFO"):
        """添加日志到显示区域"""
        # 根据日志级别设置颜色
        color = {
            "INFO": ft.colors.GREY_800,
            "ERROR": ft.colors.RED_600,
            "WARNING": ft.colors.ORANGE_600
        }.get(level, ft.colors.GREY_800)
        
        # 创建日志文本
        log_text = ft.Text(
            value=message,
            size=12,
            color=color,
            font_family="楷体",
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS
        )
        
        # 添加到日志显示区域
        self.log_display.controls.append(log_text)
        
        # 保持最新的50条日志
        if len(self.log_display.controls) > 50:
            self.log_display.controls.pop(0)
        
        self.page.update()

    def update_status(self, message):
        """更新状态显示"""
        self.progress_text.value = message
        self.add_log(message)  # 添加到日志显示
        self.page.update()
        self.logger.info(message)

    def show_error(self, message):
        """显示错误消息"""
        self.page.show_snack_bar(
            ft.SnackBar(
                content=ft.Text(message, color="white"),
                bgcolor=ft.colors.ERROR,
                action="关闭"
            )
        )
        self.progress_text.value = "处理出错"
        self.task_progress_text.value = ""
        self.add_log(message, "ERROR")  # 添加错误日志
        self.page.update()
        self.logger.error(message)

    def update_progress(self, current, total, message=""):
        """更新进度显示"""
        if total > 0:
            self.progress_bar.value = current / total
        else:
            self.progress_bar.value = 0
        self.progress_text.value = message
        if current < total:
            progress_msg = f"进度：{current}/{total} ({(current/total*100):.1f}%)"
            self.task_progress_text.value = progress_msg
            self.add_log(f"{message} - {progress_msg}")  # 添加进度日志
        else:
            self.task_progress_text.value = ""
            self.add_log(message)  # 添加完成日志
        self.page.update()

    def create_control_section(self):
        """创建控制按钮部分"""
        self.start_button = ft.ElevatedButton(
            text="开始处理",
            on_click=self.start_processing,
            style=ft.ButtonStyle(
                padding=ft.padding.all(20),
                shape=ft.RoundedRectangleBorder(radius=8)
            ),
            width=200,  # 设置按钮宽度
            height=50   # 设置按钮高度
        )
        
        return ft.Card(
            content=ft.Container(
                content=self.start_button,
                padding=ft.padding.all(20),
                alignment=ft.alignment.center
            ),
            elevation=5  # 添加阴影效果
        )

    def select_folder(self, e):
        """选择文件夹"""
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path = folder
            self.folder_text.value = folder
            self.page.update()
            self.logger.info(f"已选择文件夹: {folder}")

    # 复选框状态变化处理
    def on_collect_changed(self, e):
        self.collect_value = e.control.value
        self.page.update()

    def on_deduplicate_changed(self, e):
        self.deduplicate_value = e.control.value
        self.page.update()

    def on_resize_changed(self, e):
        self.resize_value = e.control.value
        # 更新输入框状态
        e.control.parent.controls[2].controls[1].disabled = not self.resize_value
        self.page.update()

    def on_face_changed(self, e):
        self.face_value = e.control.value
        if self.face_value:
            self.face_only_value = False
            self.face_features_value = False
        self._update_face_controls(e.control.parent)

    def on_face_only_changed(self, e):
        self.face_only_value = e.control.value
        if self.face_only_value:
            self.face_value = False
            self.face_features_value = False
        self._update_face_controls(e.control.parent)

    def on_face_features_changed(self, e):
        self.face_features_value = e.control.value
        if self.face_features_value:
            self.face_value = False
            self.face_only_value = False
        self._update_face_controls(e.control.parent)

    def _update_face_controls(self, parent):
        # 更新下拉框状态
        has_face_task = any([self.face_value, self.face_only_value, self.face_features_value])
        parent.controls[4].controls[1].disabled = not has_face_task
        self.page.update()

    def on_caption_brief_changed(self, e):
        self.caption_brief_value = e.control.value
        if self.caption_brief_value:
            self.caption_normal_value = False
            self.caption_detailed_value = False
        self.page.update()

    def on_caption_normal_changed(self, e):
        self.caption_normal_value = e.control.value
        if self.caption_normal_value:
            self.caption_brief_value = False
            self.caption_detailed_value = False
        self.page.update()

    def on_caption_detailed_changed(self, e):
        self.caption_detailed_value = e.control.value
        if self.caption_detailed_value:
            self.caption_brief_value = False
            self.caption_normal_value = False
        self.page.update()

    def start_processing(self, e):
        """开始处理"""
        if not self.folder_path:
            self.show_error("请先选择图片文件夹！")
            return
        
        # 获取任务选项
        tasks = {
            'collect': self.collect_value,
            'deduplicate': self.deduplicate_value,
            'resize': self.resize_value,
            'face': any([self.face_value, self.face_only_value, self.face_features_value]),
            'caption': any([self.caption_brief_value, self.caption_normal_value, self.caption_detailed_value])
        }
        
        if not any(tasks.values()):
            self.show_error("请至少选择一个任务！")
            return
        
        # 获取处理参数
        try:
            max_dimension = int(self.max_size_input.value) if self.resize_value else None
            face_size = 1024 if self.face_size_dropdown.value == "1024×1024" else 512
            face_only = self.face_only_value
            face_features = self.face_features_value
            
            # 获取描述详细程度
            if self.caption_brief_value:
                detail_level = 'brief'
            elif self.caption_normal_value:
                detail_level = 'normal'
            else:
                detail_level = 'detailed'
                
        except ValueError:
            self.show_error("请输入有效的尺寸数值！")
            return
        
        # 禁用界面
        self.disable_ui()
        
        # 创建处理线程
        self.processing_thread = threading.Thread(
            target=self.process_images,
            args=(tasks, max_dimension, face_size, face_only, face_features, detail_level)
        )
        self.processing_thread.start()

    def process_images(self, tasks, max_dimension, face_size, face_only, face_features, detail_level):
        """处理图片的线程函数"""
        try:
            total_tasks = sum(1 for x in tasks.values() if x)
            current_task = 0
            
            # 文件收集
            if tasks['collect']:
                current_task += 1
                self.update_status(f"[{current_task}/{total_tasks}] 正在收集文件...")
                processor = FileProcessor(self.folder_path)
                processor.collect_files()
            
            # 删除重复
            if tasks['deduplicate']:
                current_task += 1
                self.update_status(f"[{current_task}/{total_tasks}] 正在删除重复图片...")
                deduplicator = ImageDeduplicator(
                    self.folder_path,
                    progress_callback=self.update_progress
                )
                deduplicator.remove_duplicates()
            
            # 调整尺寸
            if tasks['resize']:
                current_task += 1
                self.update_status(f"[{current_task}/{total_tasks}] 正在调整图片尺寸...")
                resizer = ImageResizer(
                    max_dimension=max_dimension,
                    progress_callback=self.update_progress
                )
                success_count, fail_count = resizer.process_directory(self.folder_path)
            
            # 人脸处理
            if tasks['face']:
                current_task += 1
                self.update_status(f"[{current_task}/{total_tasks}] 正在处理人脸...")
                face_processor = FaceProcessor(
                    target_size=face_size,
                    face_only=face_only,
                    face_features=face_features,
                    progress_callback=self.update_progress
                )
                face_processor.process_folder(self.folder_path)
            
            # 生成描述
            if tasks['caption']:
                current_task += 1
                self.update_status(f"[{current_task}/{total_tasks}] 正在生成图片描述...")
                if not torch.cuda.is_available():
                    self.show_error("图片描述功能需要GPU支持，当前设备未检测到GPU")
                    return
                try:
                    captioner = ImageCaptioner(
                        detail_level,
                        progress_callback=self.update_progress
                    )
                    captioner.process_folder(self.folder_path)
                except Exception as e:
                    self.logger.error(f"图片描述生成失败: {str(e)}")
                    self.show_error(f"图片描述生成失败：\n{str(e)}")
                    return
            
            self.update_status("处理完成！")
            self.update_progress(1, 1, "全部任务处理完成")
            
        except Exception as e:
            self.logger.error(f"处理过程出错: {str(e)}", exc_info=True)
            self.show_error(f"处理过程中发生错误：\n{str(e)}")
        finally:
            self.enable_ui()

    def disable_ui(self):
        """禁用界面"""
        self.start_button.disabled = True
        self.start_button.text = "处理中..."
        self.progress_bar.value = None
        self.progress_text.value = "准备处理..."
        self.task_progress_text.value = ""
        
        # 禁用所有复选框
        for container in self.task_containers:
            for control in container.content.controls:
                if isinstance(control, ft.Checkbox):
                    control.disabled = True
        
        self.page.update()

    def enable_ui(self):
        """启用界面"""
        self.start_button.disabled = False
        self.start_button.text = "开始处理"
        self.progress_bar.value = 0
        self.progress_text.value = "准备就绪"
        self.task_progress_text.value = ""
        
        # 启用所有复选框
        for container in self.task_containers:
            for control in container.content.controls:
                if isinstance(control, ft.Checkbox):
                    control.disabled = False
        
        self.page.update()

    # ... 继续添加其他方法 ... 