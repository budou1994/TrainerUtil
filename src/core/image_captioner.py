import torch
from PIL import Image
from pathlib import Path
import logging
from transformers import AutoProcessor, LlavaForConditionalGeneration

class ImageCaptioner:
    def __init__(self, detail_level='normal', progress_callback=None):
        self.detail_level = detail_level
        self.logger = logging.getLogger("image_captioner")
        self.progress_callback = progress_callback
        
        # 模型配置
        self.model_name = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
        self.models_dir = Path("models")
        
        # 检查GPU可用性
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.logger.warning("未检测到GPU，无法执行图片描述任务")
            raise RuntimeError("图片描述功能需要GPU支持，当前设备未检测到GPU")
        
        # 定义提示词模板
        self.prompts = {
            'brief': [
                "List the key elements and main subjects in this image using keywords.",
                "Focus on the most important visual elements.",
                "Keep the description concise and use simple terms."
            ],
            'normal': [
                "Describe the main scene and primary subjects in this image.",
                "Focus on what's happening and who/what is in the image.",
                "Keep the description clear and straightforward."
            ],
            'detailed': [
                "Provide a comprehensive and in-depth description of this image.",
                "Include all aspects: setting, subjects, actions, emotions, colors, lighting, composition.",
                "Describe subtle details, artistic elements, atmosphere, and any notable features.",
                "Analyze the visual storytelling and overall impact of the image."
            ]
        }
        
        # 生成参数
        self.generation_params = {
            'brief': {
                'max_new_tokens': 100,
                'temperature': 0.3,
                'top_p': 0.9,
                'repetition_penalty': 1.2,
                'num_beams': 2,
                'length_penalty': 1.0
            },
            'normal': {
                'max_new_tokens': 150,
                'temperature': 0.5,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'num_beams': 3,
                'length_penalty': 1.0
            },
            'detailed': {
                'max_new_tokens': 500,
                'temperature': 0.8,
                'top_p': 0.95,
                'repetition_penalty': 1.0,
                'num_beams': 5,
                'length_penalty': 1.5
            }
        }
        
        self._init_model()

    def _init_model(self):
        """检查并加载JoyCaption模型"""
        try:
            model_path = self.models_dir / "joycaption"
            if not model_path.exists():
                self.logger.info("正在下载JoyCaption模型...")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.processor.save_pretrained(model_path)
                self.model.save_pretrained(model_path)
                self.logger.info("JoyCaption模型下载完成")
            else:
                self.logger.info("加载本地JoyCaption模型...")
                self.processor = AutoProcessor.from_pretrained(str(model_path))
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"加载JoyCaption模型失败: {str(e)}")
            raise

    def _generate_caption(self, image_path):
        """为单张图片生成描述"""
        try:
            # 加载图片
            image = Image.open(image_path)
            
            # 构建提示词和对话
            selected_prompts = self.prompts.get(self.detail_level, self.prompts['normal'])
            prompt = " ".join(selected_prompts)
            
            conversation = [
                {
                    "role": "system",
                    "content": "You are a professional image captioner with expertise in visual analysis. "
                              "Provide descriptions that match the requested level of detail while maintaining accuracy."
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            
            conversation_string = self.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 处理输入
            inputs = self.processor(
                text=conversation_string,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
            
            # 获取生成参数
            gen_params = self.generation_params.get(self.detail_level, 
                                               self.generation_params['normal'])
            
            # 生成描述
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **gen_params,
                    do_sample=True
                )
            
            # 解码输出
            caption = self.processor.tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()
            
            return caption
            
        except Exception as e:
            self.logger.error(f"生成描述失败: {str(e)}")
            raise

    def _save_caption(self, image_path, caption):
        """保存图片描述到文本文件"""
        output_path = Path(image_path).with_suffix('.txt')
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(caption)

    def process_folder(self, folder_path):
        """处理文件夹中的所有图片"""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"文件夹不存在: {folder_path}")

        # 收集所有jpg和png图片
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(list(folder.rglob(ext)))
        
        total_files = len(image_files)
        
        if total_files == 0:
            self.logger.warning(f"在 {folder_path} 中未找到支持的图片文件(jpg/png)")
            return
        
        if self.progress_callback:
            self.progress_callback(0, total_files, "开始生成图片描述...")
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                caption = self._generate_caption(str(image_path))
                self._save_caption(image_path, caption)
                self.logger.info(f"生成描述: {image_path.name} - {caption[:50]}...")
                
                if self.progress_callback:
                    self.progress_callback(
                        idx,
                        total_files,
                        f"处理图片: {image_path.name}"
                    )
                    
            except Exception as e:
                self.logger.error(f"处理图片失败 {image_path}: {str(e)}")
                
        if self.progress_callback:
            self.progress_callback(
                total_files,
                total_files,
                f"描述生成完成 - 共处理: {total_files}张图片"
            ) 