from api_interface.config import (
    is_share,
    webui_port_main
)
import gradio as gr
from api_interface.interface_model import create_model_app
from api_interface.interface_tts import create_tts_app
from api_interface.interface_dataset import create_dataset_app
from api_interface.interface_model_training import create_model_training_app
from api_interface.interface_uvr import create_uvr_app


# ===== 主应用: 组合所有模块 =====
def create_main_app():
    """创建主应用，组合所有模块"""
    
    # 创建模型管理模块
    model_manange_interface = create_model_app()
    uvr_interface = create_uvr_app()
    dataset_interface = create_dataset_app()
    check_sovits_train_status_interface, sovits_training_interface, check_gpt_train_status_interface, gpt_training_interface = create_model_training_app()
    model_module = gr.TabbedInterface(
        [model_manange_interface,uvr_interface,dataset_interface, check_sovits_train_status_interface, sovits_training_interface, check_gpt_train_status_interface, gpt_training_interface],
        ["当前模型查询", "UVR人声分离", "数据集预处理", "SoVITS模型训练状态检查","SoVITS模型训练", "GPT模型训练状态检查","GPT模型训练"],
    )
    # 创建TTS模块
    tts_module = create_tts_app()
    # 使用 TabbedInterface 组合
    main_app = gr.TabbedInterface(
        [model_module,tts_module],
        ["GPT-SoVITS模型管理","GPT-SoVITS TTS服务"],
        title="GPT-SoVITS API服务"
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
