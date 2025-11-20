import gradio as gr
import numpy as np
import torch
import librosa
import traceback
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import io
import base64
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

def get_tts_wav_api(
    ref_wav_file: gr.Audio,  # 改为接受 gradio Audio 类型
    prompt_text: str,
    prompt_language: str,
    text: str,
    text_language: str,
    how_to_cut: str = "不切",
    top_k: int = 20,
    top_p: float = 0.6,
    temperature: float = 0.6,
    ref_free: bool = False,
    speed: float = 1.0,
    if_freeze: bool = False,
    inp_refs: Optional[List] = None,
    sample_steps: int = 8,
    if_sr: bool = False,
    pause_second: float = 0.3
) -> Dict[str, Any]:
    """
    TTS 音频生成 API 版本
    输入音频文件，输出音频文件和相关信息
    
    Args:
        ref_wav_file: 参考音频文件 (gradio.Audio 对象)
        prompt_text: 提示文本
        prompt_language: 提示文本语言
        text: 目标文本
        text_language: 目标文本语言
        how_to_cut: 文本切割方式
        top_k: top_k 参数
        top_p: top_p 参数
        temperature: 温度参数
        ref_free: 是否无参考
        speed: 语速
        if_freeze: 是否冻结
        inp_refs: 额外参考文件
        sample_steps: 采样步数
        if_sr: 是否超分辨率
        pause_second: 停顿秒数
    
    Returns:
        Dict: 包含音频文件和元数据的字典
    """
    
    # 初始化缓存（如果是全局的，这里可以移除）
    cache = {}
    
    # 参数验证
    if not ref_wav_file:
        return {
            "status": "error",
            "message": "请上传参考音频",
            "audio": None,
            "sample_rate": None
        }
    
    if not text:
        return {
            "status": "error", 
            "message": "请填入推理文本",
            "audio": None,
            "sample_rate": None
        }
    
    try:
        # 从 gradio.Audio 对象获取音频数据
        if isinstance(ref_wav_file, tuple):
            # gradio.Audio 返回 (sample_rate, audio_data)
            ref_sample_rate, ref_audio_data = ref_wav_file
            ref_wav_path = save_temp_audio(ref_audio_data, ref_sample_rate)
        else:
            # 如果是文件路径
            ref_wav_path = ref_wav_file
        
        from GPT_SoVITS.inference_webui import get_tts_wav
        # 这里我们重构处理逻辑，但保持核心算法
        opt_sr, audio_data = get_tts_wav(
            ref_wav_path=ref_wav_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            text=text,
            text_language=text_language,
            how_to_cut=how_to_cut,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            ref_free=ref_free,
            speed=speed,
            if_freeze=if_freeze,
            inp_refs=inp_refs,
            sample_steps=sample_steps,
            if_sr=if_sr,
            pause_second=pause_second
        )
        
        # 清理临时文件
        if isinstance(ref_wav_file, tuple) and os.path.exists(ref_wav_path):
            os.remove(ref_wav_path)
        
        return audio_data, {
            "status": "success",
            "message": "音频生成成功",
            "audio": (opt_sr, audio_data),
            "sample_rate": opt_sr,
            "audio_duration": len(audio_data) / opt_sr,
            "text_processed": text,
            "timestamp": gr.utils.get_current_time()
        }
        
    except Exception as e:
        return None,{
            "status": "error",
            "message": f"音频生成失败: {str(e)}",
            "audio": None,
            "sample_rate": None,
            "traceback": traceback.format_exc()
        }


def save_temp_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    保存音频数据到临时文件
    
    Args:
        audio_data: 音频数据数组
        sample_rate: 采样率
    
    Returns:
        str: 临时文件路径
    """
    import soundfile as sf
    
    # 创建临时文件
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"temp_ref_audio_{os.getpid()}_{id(audio_data)}.wav")
    
    # 保存音频
    sf.write(temp_file, audio_data, sample_rate)
    
    return temp_file


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    标准化音频数据
    
    Args:
        audio_data: 原始音频数据
    
    Returns:
        np.ndarray: 标准化后的音频数据
    """
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32767.0
    elif audio_data.dtype == np.float64:
        audio_data = audio_data.astype(np.float32)
    
    # 限制幅度在 [-1, 1] 范围内
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val
    
    return audio_data


# ===== 创建纯 API 接口 =====

def create_tts_app():
    """创建 TTS API 接口"""
    
    # 定义语言选项
    language_choices = ["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"]
    cut_method_choices = ["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"]
    
    # 创建接口
    tts_interface = gr.Interface(
        fn=get_tts_wav_api,
        inputs=[
            gr.Audio(
                label="参考音频",
                type="filepath"
            ),
            gr.Textbox(
                label="提示文本",
                placeholder="请输入参考音频对应的文本...",
            ),
            gr.Dropdown(
                choices=language_choices,
                label="提示文本语言",
                value="中文"
            ),
            gr.Textbox(
                label="目标文本", 
                placeholder="请输入要合成的文本...",
                lines=3,
            ),
            gr.Dropdown(
                choices=language_choices,
                label="目标文本语言", 
                value="中文"
            ),
            gr.Dropdown(
                choices=cut_method_choices,
                label="文本切割方式",
                value="不切",
            ),
            gr.Slider(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                label="采样时的top-k参数"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.6,
                step=0.1,
                label="采样时的top-p参数",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="采样温度",
            ),
            gr.Checkbox(
                label="无参考模式",
                value=False,
                info="是否不使用参考音频特征"
            ),
            gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="语速",
            ),
            gr.Checkbox(
                label="冻结缓存",
                value=False,
                info="是否使用缓存加速生成"
            ),
            gr.File(
                label="额外参考文件",
                file_count="multiple",
                visible=False
            ),
            gr.Slider(
                minimum=1,
                maximum=20,
                value=8,
                step=1,
                label="CFM采样步数",
            ),
            gr.Checkbox(
                label="超分辨率",
                value=False,
                info="是否启用音频超分辨率"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.3,
                step=0.1,
                label="句子间的停顿时间（秒）",
            )
        ],
        outputs=[
            gr.Audio(label="生成音频", type="numpy"),
            gr.JSON(label="生成信息")
        ],
        title="GPT-SoVITS TTS API 服务",
        description="文本到语音合成 API 接口，支持多语言和声音克隆",
        api_name="tts_generate"
    )
    
    return tts_interface


def create_simple_tts_api():
    """创建简化版 TTS API"""
    
    def simple_tts_api(
        ref_audio: gr.Audio,
        text: str,
        language: str = "中文",
        speed: float = 1.0
    ):
        """简化版 TTS API"""
        return get_tts_wav_api(
            ref_wav_file=ref_audio,
            prompt_text="",  # 自动从参考音频提取或使用默认
            prompt_language=language,
            text=text,
            text_language=language,
            speed=speed
        )
    
    simple_interface = gr.Interface(
        fn=simple_tts_api,
        inputs=[
            gr.Audio(label="参考音频", type="filepath"),
            gr.Textbox(label="合成文本", placeholder="输入要合成的文本...", lines=2),
            gr.Dropdown(
                choices=["中文", "英文", "日文", "中英混合"],
                label="语言",
                value="中文"
            ),
            gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="语速")
        ],
        outputs=[
            gr.Audio(label="生成音频"),
            gr.JSON(label="生成状态")
        ],
        title="简化版 TTS API",
        description="快速文本到语音合成接口",
        api_name="simple_tts"
    )
    
    return simple_interface


def create_batch_tts_api():
    """创建批量 TTS API"""
    
    def batch_tts_api(
        ref_audio: gr.Audio,
        texts: str,  # JSON 字符串或每行一个文本
        language: str = "中文",
        speed: float = 1.0
    ):
        """批量 TTS 生成"""
        try:
            # 解析文本输入
            if texts.strip().startswith('['):
                # JSON 格式
                import json
                text_list = json.loads(texts)
            else:
                # 每行一个文本
                text_list = [line.strip() for line in texts.split('\n') if line.strip()]
            
            results = []
            for i, text in enumerate(text_list):
                result = get_tts_wav_api(
                    ref_wav_file=ref_audio,
                    prompt_text="",
                    prompt_language=language,
                    text=text,
                    text_language=language,
                    speed=speed
                )
                results.append({
                    "index": i,
                    "text": text,
                    "result": result
                })
            
            return {
                "status": "success",
                "total": len(results),
                "results": results,
                "timestamp": gr.utils.get_current_time()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"批量处理失败: {str(e)}",
                "timestamp": gr.utils.get_current_time()
            }
    
    batch_interface = gr.Interface(
        fn=batch_tts_api,
        inputs=[
            gr.Audio(label="参考音频", type="filepath"),
            gr.Textbox(
                label="批量文本",
                placeholder='["文本1", "文本2", ...] 或 每行一个文本',
                lines=5,
            ),
            gr.Dropdown(
                choices=["中文", "英文", "日文"],
                label="语言",
                value="中文"
            ),
            gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="语速")
        ],
        outputs=gr.JSON(label="批量生成结果"),
        title="批量 TTS API",
        description="批量文本到语音合成接口",
        api_name="batch_tts"
    )
    
    return batch_interface


# ===== 创建完整的 API 服务 =====

def create_tts_api_service():
    """创建完整的 TTS API 服务"""
    
    with gr.Blocks(title="GPT-SoVITS TTS API 服务", theme=gr.themes.Soft()) as api_service:
        
        gr.Markdown("# 🎙️ GPT-SoVITS TTS API 服务")
        gr.Markdown("提供完整的文本到语音合成 API 接口，支持声音克隆和多种语言")
        
        with gr.Tabs():
            # 标准 API
            with gr.TabItem("🔧 标准 TTS API"):
                tts_interface = create_tts_api_interface()
                # 这里需要将界面组件添加到 Blocks 中
                # 由于 Interface 不能直接嵌入，我们需要重新创建组件
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 标准 TTS 参数")
                        ref_audio = gr.Audio(label="参考音频", type="filepath")
                        prompt_text = gr.Textbox(label="提示文本")
                        prompt_lang = gr.Dropdown(
                            choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"],
                            label="提示文本语言",
                            value="中文"
                        )
                        target_text = gr.Textbox(label="目标文本", lines=3)
                        target_lang = gr.Dropdown(
                            choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"],
                            label="目标文本语言", 
                            value="中文"
                        )
                        cut_method = gr.Dropdown(
                            choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"],
                            label="文本切割方式",
                            value="不切"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 高级参数")
                        top_k = gr.Slider(1, 100, value=20, step=1, label="Top-K")
                        top_p = gr.Slider(0.1, 1.0, value=0.6, step=0.1, label="Top-P")
                        temperature = gr.Slider(0.1, 2.0, value=0.6, step=0.1, label="Temperature")
                        ref_free = gr.Checkbox(label="无参考模式", value=False)
                        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="语速")
                        if_freeze = gr.Checkbox(label="冻结缓存", value=False)
                        sample_steps = gr.Slider(1, 20, value=8, step=1, label="采样步数")
                        if_sr = gr.Checkbox(label="超分辨率", value=False)
                        pause_second = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="停顿时间")
                
                with gr.Row():
                    submit_btn = gr.Button("生成语音", variant="primary")
                
                with gr.Row():
                    audio_output = gr.Audio(label="生成音频", interactive=False)
                    json_output = gr.JSON(label="生成信息")
                
                # 绑定事件
                submit_btn.click(
                    fn=get_tts_wav_api,
                    inputs=[
                        ref_audio, prompt_text, prompt_lang, target_text, target_lang,
                        cut_method, top_k, top_p, temperature, ref_free, speed,
                        if_freeze, gr.Textbox(visible=False), sample_steps, if_sr, pause_second
                    ],
                    outputs=[audio_output, json_output],
                    api_name="tts_generate"
                )
            
            # 简化 API
            with gr.TabItem("⚡ 快速 TTS API"):
                simple_interface = create_simple_tts_api()
                # 同样需要重新创建简化界面组件...
            
            # 批量 API
            with gr.TabItem("📦 批量 TTS API"):
                batch_interface = create_batch_tts_api()
                # 重新创建批量界面组件...
        
        # API 文档
        gr.Markdown("---")
        gr.Markdown("## 📚 API 文档")
        gr.Markdown("""
        ### 标准 TTS API
        **端点**: `/api/call/tts_generate`
        
        **参数**:
        ```json
        {
            "data": [
                "参考音频文件", 
                "提示文本", 
                "提示语言", 
                "目标文本", 
                "目标语言",
                "切割方式",
                20, 0.6, 0.6, false, 1.0, false, null, 8, false, 0.3
            ]
        }
        ```
        
        **返回**:
        ```json
        {
            "status": "success",
            "audio": "音频数据",
            "sample_rate": 24000,
            "audio_duration": 5.2,
            "text_processed": "处理后的文本"
        }
        ```
        """)
    
    return api_service


# ===== 客户端调用示例 =====

class TTSAPIClient:
    """TTS API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.session = requests.Session() if 'requests' in globals() else None
    
    def generate_tts(self, audio_file: str, text: str, **kwargs) -> Dict[str, Any]:
        """
        生成 TTS 音频
        
        Args:
            audio_file: 参考音频文件路径
            text: 目标文本
            **kwargs: 其他参数
        
        Returns:
            Dict: 生成结果
        """
        if not self.session:
            import requests
            self.session = requests.Session()
        
        # 准备参数
        params = {
            "data": [
                audio_file,
                kwargs.get("prompt_text", ""),
                kwargs.get("prompt_language", "中文"),
                text,
                kwargs.get("text_language", "中文"),
                kwargs.get("how_to_cut", "不切"),
                kwargs.get("top_k", 20),
                kwargs.get("top_p", 0.6),
                kwargs.get("temperature", 0.6),
                kwargs.get("ref_free", False),
                kwargs.get("speed", 1.0),
                kwargs.get("if_freeze", False),
                kwargs.get("inp_refs", None),
                kwargs.get("sample_steps", 8),
                kwargs.get("if_sr", False),
                kwargs.get("pause_second", 0.3)
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/call/tts_generate",
                json=params,
                timeout=300  # 5分钟超时
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}


# ===== 启动服务 =====

if __name__ == "__main__":
    # 创建并启动 API 服务
    api_service = create_tts_app()
    
    api_service.queue(
        concurrency_count=2,  # 根据GPU内存调整
        max_size=10,
        api_open=True
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True,
        enable_api=True,
        quiet=False
    )