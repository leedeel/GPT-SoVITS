import gradio as gr
from api_interface.config import get_weights_names


# ===== 模块1: 模型服务 =====
def create_model_app():
    """创建模型管理应用模块"""
    
    def list_models():
        SoVITS_names, GPT_names = get_weights_names()
        return {"SoVITS": SoVITS_names, "GPT": GPT_names}
    
    model_interface = gr.Interface(
        fn= list_models,
        inputs=[],
        outputs=gr.JSON(),
        api_name="model_list",
        title="GPT-SoVITS模型API服务"
    )
    
    return model_interface
