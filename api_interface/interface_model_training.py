import gradio as gr
from api_interface.model_training.gpt_fn import train_gpt, check_gpt_train_status
from api_interface.model_training.sovits_fn import train_sovits, check_sovits_train_status


def create_model_training_app():
    """创建模型训练应用模块"""
    version_choices = ["v1", "v2", "v4", "v2Pro", "v2ProPlus"]
    
    check_sovits_train_status_interface = gr.Interface(
        fn=check_sovits_train_status,
        inputs=[],
        outputs=gr.JSON(label="SoVITS模型训练状态"),
        title="SoVITS模型训练状态",
        api_name="check_sovits_train_status_api"
    )
    sovits_training_interface = gr.Interface(
        fn=train_sovits,
        inputs=[
            gr.Radio(
                label="训练模型的版本",
                value=version_choices[-1],
                choices=version_choices,
                scale=5,
            ),
            gr.Textbox(label="实验名称")],
        outputs=gr.JSON(label="SoVITS模型训练结果"),
        title="SoVITS模型训练",
        api_name="train_sovits_api"
    )
    
    check_gpt_train_status_interface = gr.Interface(
        fn=check_gpt_train_status,
        inputs=[],
        outputs=gr.JSON(label="GPT模型训练状态"),
        title="GPT模型训练状态",
        api_name="check_gpt_train_status_api"
    )
    gpt_training_interface = gr.Interface(
        fn=train_gpt,
        inputs=[
            gr.Radio(
                label="训练模型的版本",
                value=version_choices[-1],
                choices=version_choices,
                scale=5,
            ),
            gr.Textbox(label="实验名称")],
        outputs=gr.JSON(label="GPT模型训练结果"),
        title="GPT模型训练",
        api_name="train_gpt_api"
    )
    return check_sovits_train_status_interface, sovits_training_interface, check_gpt_train_status_interface, gpt_training_interface


if __name__ == "__main__":
    check_sovits_train_status_interface, sovits_training_interface, check_gpt_train_status_interface, gpt_training_interface = create_model_training_app()
    model_training_module = gr.TabbedInterface(
        [check_sovits_train_status_interface, sovits_training_interface, check_gpt_train_status_interface, gpt_training_interface],
        ["SoVITS模型训练状态检查","SoVITS模型训练", "GPT模型训练状态检查","GPT模型训练"],
        title="模型训练"
    )
    model_training_module.launch()