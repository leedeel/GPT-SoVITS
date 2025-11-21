from config import (
    is_share,
    webui_port_main
)
import gradio as gr
from api_interface.interface_model import create_model_app
from api_interface.interface_tts import create_tts_app


# ===== 主应用: 组合所有模块 =====
def create_main_app():
    """创建主应用，组合所有模块"""
    
    # 创建各个模块
    model_module = create_model_app()
    tts_module = create_tts_app()
    
    # 使用 TabbedInterface 组合
    main_app = gr.TabbedInterface(
        [model_module,tts_module],
        ["GPT-SoVITS模型API服务","GPT-SoVITS TTS API服务"],
        title="模块化应用集合"
    )
    
    return main_app


if __name__ == "__main__":
    app = create_main_app()
    app.queue().launch(  # concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_main,
        # quiet=True,
    )
