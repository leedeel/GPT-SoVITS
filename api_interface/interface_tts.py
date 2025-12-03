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
from GPT_SoVITS.inference_webui_fn import i18n,get_tts_wav


def cleanup_memory():
    """清理内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_tts_wav_api(
    sovits_path: str,
    gpt_path: str,
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
        sovits_path: SoVITS模型路径
        gpt_path: GPT模型路径
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
        
        # 这里我们重构处理逻辑，但保持核心算法
        opt_sr,audio_data = next(get_tts_wav(
            sovits_path=sovits_path,
            gpt_path=gpt_path,
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
        ))
        
        # 清理临时文件
        if isinstance(ref_wav_file, tuple) and os.path.exists(ref_wav_path):
            os.remove(ref_wav_path)
        # 清理所有内存
        cleanup_memory()
        return (opt_sr,audio_data), {
            "status": "success",
            "message": "音频生成成功",
            "sample_rate": opt_sr,
            "audio_duration": len(audio_data) / opt_sr,
            "text_processed": text,
            "text_processed": text
        }
        
    except Exception as e:
        return None,{
            "status": "error",
            "message": f"音频生成失败: {str(e)}",
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



def create_tts_app():
    """创建 TTS API 接口"""
    # 定义语言选项
    language_choices = ["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"]
    cut_method_choices = ["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"]
    
    # 初始化模型列表
    SoVITS_names, GPT_names = get_weights_names()
    
    # 刷新模型列表的函数
    def refresh_models():
        """刷新模型列表"""
        nonlocal SoVITS_names, GPT_names
        SoVITS_names, GPT_names = get_weights_names()
        
        # 返回更新后的下拉菜单选项
        return {"choices": SoVITS_names, "__type__": "update"}, {"choices": GPT_names,"__type__": "update",}

    
    # 使用 Blocks 替代 Interface，以获得更大的布局灵活性
    with gr.Blocks(title="GPT-SoVITS TTS API 服务") as tts_interface:
        gr.Markdown("# GPT-SoVITS TTS API 服务")
        gr.Markdown("文本到语音合成 API 接口，支持多语言和声音克隆")
        with gr.Row():
            # 输入区域
            with gr.Column(scale=4):
                # 模型选择行
                with gr.Row(scale=2):
                    # SoVITS 模型选择
                    sovits_dropdown = gr.Dropdown(
                        label=i18n("SoVITS模型列表"),
                        choices=SoVITS_names,
                        value=SoVITS_names[0] if SoVITS_names else None,
                        interactive=True,
                        scale=6,
                        filterable=True,
                        info="选择音色模型"
                    )
                    
                    # GPT 模型选择
                    gpt_dropdown = gr.Dropdown(
                        label=i18n("GPT模型列表"),
                        choices=GPT_names,
                        value=GPT_names[0] if GPT_names else None,
                        interactive=True,
                        scale=6,
                        filterable=True,
                        info="选择文本模型"
                    )
                    
                    # 刷新按钮
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button(
                            "🔄 刷新模型",
                            variant="secondary",
                            size="sm",
                            min_width=80
                        )
                        # 显示模型数量
                        model_count_info = gr.Markdown(
                            f"SoVITS: {len(SoVITS_names)}个 | GPT: {len(GPT_names)}个"
                        )
                
                # 其他输入参数
                with gr.Row(scale=4):
                    ref_audio = gr.Audio(
                        label="参考音频",
                        type="filepath",
                        scale=6
                    )
                    
                    prompt_text = gr.Textbox(
                        label="提示文本",
                        placeholder="请输入参考音频对应的文本...",
                        scale=6
                    )
                
                with gr.Row(scale=1):
                    prompt_language = gr.Dropdown(
                        choices=language_choices,
                        label="提示文本语言",
                        value="中文",
                        scale=3
                    )
                    
                    text_input = gr.Textbox(
                        label="目标文本", 
                        placeholder="请输入要合成的文本...",
                        lines=3,
                        scale=6
                    )
                    
                    text_language = gr.Dropdown(
                        choices=language_choices,
                        label="目标文本语言", 
                        value="中文",
                        scale=3
                    )
                with gr.Accordion("更多TTS设置", open=False):
                    with gr.Row():
                        cut_method = gr.Dropdown(
                            choices=cut_method_choices,
                            label="文本切割方式",
                            value="不切",
                            scale=3
                        )
                        
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=20,
                            step=1,
                            label="采样top-k",
                            scale=3
                        )
                        
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.6,
                            step=0.1,
                            label="采样top-p",
                            scale=3
                        )
                        
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.6,
                            step=0.1,
                            label="采样温度",
                            scale=3
                        )
                    
                    with gr.Row():
                        use_ref_audio = gr.Checkbox(
                            label="无参考模式",
                            value=False,
                            info="是否不使用参考音频特征",
                            scale=2
                        )
                        
                        speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="语速",
                            scale=2
                        )
                        
                        use_cache = gr.Checkbox(
                            label="冻结缓存",
                            value=False,
                            info="是否使用缓存加速生成",
                            scale=2
                        )
                        
                        extra_ref_files = gr.File(
                            label="额外参考文件",
                            file_count="multiple",
                            visible=False,
                            scale=2
                        )
                    
                    with gr.Row():
                        cfm_steps = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=8,
                            step=1,
                            label="CFM采样步数",
                            scale=2
                        )
                        
                        super_resolution = gr.Checkbox(
                            label="超分辨率",
                            value=False,
                            info="是否启用音频超分辨率",
                            scale=2
                        )
                        
                        pause_duration = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="句子间停顿时间(秒)",
                            scale=2
                        )
                
                # 提交按钮和输出区域
                with gr.Row(scale=1):
                    submit_btn = gr.Button("开始合成", variant="primary", scale=2)
            # 输出区域
            with gr.Column(scale=1):
                with gr.Row(scale=2):
                    audio_output = gr.Audio(label="生成音频", type="numpy", scale=12)
                with gr.Row(scale=1):
                    json_output = gr.JSON(label="生成信息" )
            
        # 刷新按钮的事件绑定
        def update_model_count():
            """更新模型数量显示"""
            return f"SoVITS: {len(SoVITS_names)}个 | GPT: {len(GPT_names)}个"
        
        refresh_btn.click(
            fn=refresh_models,
            inputs=[],
            outputs=[sovits_dropdown, gpt_dropdown]
        ).then(
            fn=update_model_count,
            inputs=[],
            outputs=[model_count_info]
        )
        submit_btn.click(
            fn=get_tts_wav_api,
            inputs=[
                sovits_dropdown, gpt_dropdown, ref_audio, prompt_text,
                prompt_language, text_input, text_language, cut_method,
                top_k, top_p, temperature, use_ref_audio,
                speed, use_cache, extra_ref_files, cfm_steps,
                super_resolution, pause_duration
            ],
            outputs=[audio_output, json_output],
            api_name="tts_generate"
        )
        
        
    return tts_interface

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