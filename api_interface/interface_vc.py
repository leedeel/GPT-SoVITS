# 废弃
import gradio as gr
import numpy as np
import traceback
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
import gc
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))
from api_interface.config import get_weights_names
from GPT_SoVITS.vc_webui_fn import i18n,get_vc_wav


def cleanup_memory():
    """清理内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_vc_wav_api(
    sovits_path:str,
    gpt_path:str,
    source_wav_file: gr.Audio,
    source_wav_text:str, 
    target_wav_file:gr.Audio,
    language:str = "中文", 
    noise_scale:float=0.5
) -> Dict[str, Any]:
    """
    VC 音频生成 API 版本
    输入音频文件，输出音频文件和相关信息
    
    Args:
        sovits_path: SoVITS模型路径
        source_wav_file: 源音频文件 (gradio.Audio 对象)
        source_wav_text: 源音频文本
        target_wav_file: 参考音频文件 (gradio.Audio 对象)
        language: 参考音频语言
        noise_scale: 噪声比例
    
    Returns:
        Dict[str, Any]: 音频生成结果
        Dict[str, Any]: 音频生成信息
    """
    # 参数验证
    if not source_wav_file:
        return {
            "status": "error",
            "message": "请上传源音频文件",
            "audio": None,
            "sample_rate": None
        }
    
    try:
        # 从 gradio.Audio 对象获取音频数据
        if isinstance(source_wav_file, tuple):
            # gradio.Audio 返回 (sample_rate, audio_data)
            source_wav_sample_rate, source_wav_audio_data = source_wav_file
            source_wav_path = save_temp_audio(source_wav_sample_rate, source_wav_audio_data)
        else:
            # 如果是文件路径
            source_wav_path = source_wav_file
            
        # 从 gradio.Audio 对象获取音频数据
        if isinstance(target_wav_file, tuple):
            # gradio.Audio 返回 (sample_rate, audio_data)
            target_wav_sample_rate, target_wav_audio_data = target_wav_file
            target_wav_path = save_temp_audio(target_wav_sample_rate, target_wav_audio_data)
        else:
            # 如果是文件路径
            target_wav_path = target_wav_file
        
        # 这里我们重构处理逻辑，但保持核心算法
        opt_sr,audio_data = next(get_vc_wav(
            sovits_path=sovits_path,
            gpt_path=gpt_path,
            source_wav_path=source_wav_path,
            source_wav_text=source_wav_text,
            target_wav_path=target_wav_path,
            language=language,
        ))
        
        # 清理临时文件
        if isinstance(target_wav_file, tuple) and os.path.exists(target_wav_path):
            os.remove(target_wav_path)
        if isinstance(source_wav_file, tuple) and os.path.exists(source_wav_path):
            os.remove(source_wav_path)
        # 清理所有内存
        cleanup_memory()
        return (opt_sr,audio_data), {
            "status": "success",
            "message": "音频克隆成功",
            "sample_rate": opt_sr,
            "audio_duration": len(audio_data) / opt_sr,
        }
        
    except Exception as e:
        return None,{
            "status": "error",
            "message": f"音频克隆失败: {str(e)}",
            "audio": None,
            "traceback": traceback.format_exc()
        }
    finally:
        cleanup_memory()


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

def create_vc_app():
    """创建 TTS API 接口"""
    # 定义语言选项
    language_choices = ["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"]
    
    # 创建接口
    SoVITS_names, GPT_names = get_weights_names()
    vc_interface = gr.Interface(
        fn=get_vc_wav_api,
        inputs=[
            gr.Dropdown(
                label=i18n("SoVITS模型列表"),
                choices=SoVITS_names,
                value=SoVITS_names[0],
                interactive=True,
                scale=14,
            ),
            gr.Dropdown(
                label=i18n("GPT模型列表"),
                choices=GPT_names,
                value=GPT_names[0],
                interactive=True,
                scale=14,
            ),
            gr.Audio(
                label="源音频文件",
                type="filepath"
            ),
            gr.Textbox(
                label="源音频文本",
                placeholder="请输入源音频文本...",
            ),
            gr.Audio(
                label="目标音色音频文件",
                type="filepath"
            ),
            gr.Dropdown(
                choices=language_choices,
                label="文本语言",
                value="中文"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="噪声比例",
            ),
        ],
        outputs=[
            gr.Audio(label="生成音频", scale=12, type="numpy"),
            gr.JSON(label="生成信息")
        ],
        title="GPT-SoVITS VC API 服务",
        description="语音克隆 API 接口，支持多语言和声音克隆",
        api_name="vc_generate"
    )
    
    return vc_interface

if __name__ == "__main__":
    # 创建并启动 API 服务
    api_service = create_vc_app()
    
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