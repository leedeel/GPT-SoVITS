import gradio as gr

def uvr_api(
    exp_name: str,
    model_name: str,
    audio_files:list[str],
    format0:str="wav")->list[dict]:
    """
    uvr 接口
    :param exp_name: 实验名称
    :param model_name: 模型名称
    :param audio_files: 音频文件列表
    :param agg: 聚合度
    :param format0: 格式, wav,flac,mp3,m4a
    :return: 
    """
    try:
        from api_interface.uvr.uvr5 import uvr
        # 人声提取激进程度
        agg =10
        # 调用 uvr 函数
        uvr_result_list = uvr(exp_name=exp_name,
                         model_name=model_name,
                         paths=audio_files,
                         agg=agg,
                         format0=format0)
        # uvr_result 是一个字典，包含了每个音频文件的处理结果
        uvr_ins_file_list:list[str] = []
        uvr_vocal_file_list:list[str] = []
        for uvr_result in uvr_result_list:
            if uvr_result["status"] == "success":
                uvr_ins_file_list.append(uvr_result["ins_path"])
                uvr_vocal_file_list.append(uvr_result["vocal_path"])
        # 返回结果
        result = {
            "status": "success",
            "message": "uvr 接口调用成功",
            "uvr_result_list": uvr_result_list
        }
        return result, uvr_ins_file_list, uvr_vocal_file_list
    except Exception as e:
        return {"status": "error", "message": str(e)},[],[]
    


def create_uvr_app():
    """创建 uvr 接口"""
    from api_interface.uvr.uvr5 import get_uvr5_names
    uvr5_names = get_uvr5_names()
    
    uvr_interface = gr.Interface(
        fn=uvr_api,
        inputs=[
            gr.Textbox(
                label="实验名称",
                placeholder="请输入实验名称...",
            ),
            gr.Dropdown(label="模型名称", choices=uvr5_names, value=uvr5_names[-1]),
            gr.Files(
                label="上传音频文件",
                file_types=["audio"],
                file_count="multiple",  # 允许多文件上传
                type="filepath"  # 返回文件路径列表
            ),
            gr.Dropdown(
                choices=["wav", "flac", "mp3", "m4a"],
                value="wav",
                label="输出格式"
            )],
        outputs=[ 
                gr.JSON(
                    label="处理结果",
                    show_label=True
                ),
                gr.Files(
                    label="提取的伴奏文件",
                    show_label=True,
                    file_count="multiple"
                ),
                gr.Files(
                    label="提取的人声文件",
                    show_label=True,
                    file_count="multiple"
                )],
        title="UVR 音频分离系统",
        description="上传音频文件进行人声和伴奏分离",
        api_name="uvr_api"
    )
    return uvr_interface

if __name__ == "__main__":
    uvr_interface = create_uvr_app()
    uvr_interface.launch()