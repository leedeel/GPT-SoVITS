import gradio as gr
from config import get_weights_names
from GPT_SoVITS.inference_webui_fn import (change_gpt_weights,change_sovits_weights)


# ===== 模块1: 模型服务 =====
def create_model_app():
    """创建模型管理应用模块"""
    
    def list_models():
        SoVITS_names, GPT_names = get_weights_names()
        return {"SoVITS": SoVITS_names, "GPT": GPT_names}
    
   
   
    with gr.Blocks(title="GPT-SoVITS模型API服务", theme=gr.themes.Soft()) as model_app:
        gr.Markdown("## 🤖 模型管理")
        with gr.Row():
            list_btn = gr.Button("列出模型")
            output = gr.JSON()
        list_btn.click(fn=list_models, outputs=output, api_name="model_list")
    
    return model_app
