import gradio as gr
from config import get_weights_names


# ===== 模块1: 模型服务 =====
def create_model_app():
    """创建模型管理应用模块"""
    
    def list_models():
        SoVITS_names, GPT_names = get_weights_names()
        return {"SoVITS": SoVITS_names, "GPT": GPT_names}
    
    def refresh_models():
        SoVITS_names, GPT_names = change_choices()
        return {"status": "refreshed", "SoVITS": SoVITS_names, "GPT": GPT_names}
    
    with gr.Blocks() as model_app:
        gr.Markdown("## 🤖 模型管理")
        
        with gr.Row():
            list_btn = gr.Button("列出模型")
            refresh_btn = gr.Button("刷新列表")
            output = gr.JSON()
        
        list_btn.click(list_models, outputs=output, api_name="model_list")
        refresh_btn.click(refresh_models, outputs=output, api_name="model_refresh")
    
    return model_app
