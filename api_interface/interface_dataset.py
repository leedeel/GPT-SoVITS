import gradio as gr
import json
from api_interface.dataset.dataset_fn import train_dataset


def process_json_with_audio(json_data:str, audio_files:list[gr.Audio])->list[dict]:
    """处理JSON数据并匹配音频文件"""
    try:
        # 解析JSON
        items = json.loads(json_data)
        
        # 匹配音频文件
        if audio_files:
            audio_paths = [f.name for f in audio_files]
            
            # 假设JSON中的顺序与上传的文件顺序对应
            for i, item in enumerate(items):
                if i < len(audio_paths):
                    item['wav_path'] = audio_paths[i]
        
        # 处理每个项目
        results = []
        for item in items:
            if 'text' in item:
                text = item['text']
                audio_path = item.get('wav_path', '无音频')
                results.append(f"文本: {text}, 音频: {audio_path}")
        
        return results
    except json.JSONDecodeError:
        return "JSON格式错误"

def train_dataset_api(version:str,
                      dataset_key:str,
                      audio_json:str,
                      audio_files:list[gr.Audio]) -> dict:
    """
    训练数据集管理API
    参数:
    dataset_key: 数据集标识符
    audio_json: 音频JSON数据
    audio_files: 音频文件列表
    返回:
    数据集管理结果
    """
    try:
        train_dataset_list = process_json_with_audio(json_data=audio_json, audio_files=audio_files)
        
        train_dataset(version=version,
                    train_dataset_list=train_dataset_list,
                    dataset_key=dataset_key)
        return {"status": "success", "message": "数据集训练成功"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_dataset_app():
    """创建数据集管理应用模块"""

    # 示例JSON数据
    example_json = """[
        {"text": "早上好，今天天气不错"},
        {"text": "下午的会议很重要"},
        {"text": "晚上去看电影"}
    ]"""
    version_choices = ["v1", "v2", "v4", "v2Pro", "v2ProPlus"]

    dataset_interface = gr.Interface(
        fn=train_dataset_api,
        inputs=[
            gr.Radio(
                label="训练模型的版本",
                value=version_choices[-1],
                choices=version_choices,
                scale=5,
            ),
            gr.Textbox(
                label="数据集标识符",
                placeholder="请输入数据集标识符...",
            ),
            gr.Textbox(
                label="音频JSON数据",
                value=example_json,
                placeholder="请输入音频JSON数据...",
                lines=10,
            ),
            gr.File(
                label="上传对应音频文件",
                file_count="multiple",
                file_types=["audio"]
            )
        ],
        outputs=gr.JSON(label="数据集管理结果"),
        api_name="train_dataset_api",
        title="GPT-SoVITS数据集管理API服务"
    )
    return dataset_interface

dataset_interface = create_dataset_app()
dataset_interface.launch()