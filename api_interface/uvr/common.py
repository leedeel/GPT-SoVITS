import torch
import os

def get_device():
    """
    获取设备
    返回:
    设备
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device

def get_tmp_dir():
    now_dir = os.getcwd()
    tmp = os.path.join(now_dir, "TEMP", "uvr5")
    os.makedirs(tmp, exist_ok=True)
    return tmp


def get_vrv5_file_name_opt_dir(root_dir:str, exp_name:str, file_name:str):    
    """
    获取vrv5存储路径
    参数:
    root_dir: 根目录
    exp_name: 实验名称
    返回:
    vrv5存储路径
    """
    vrv5_dir = os.path.join(root_dir, exp_name, f"vrv5_{file_name}")
    os.makedirs(vrv5_dir, exist_ok=True)
    return vrv5_dir


def find_files_with_prefix(folder_path, prefix):
    """
    在指定文件夹中查找所有以指定前缀开头的文件。

    Args:
        folder_path (str): 要搜索的文件夹路径。
        prefix (str): 要查找的文件名前缀。

    Returns:
        list: 匹配到的文件名的列表。
    """
    matching_files = []
    try:
        for filename in os.listdir(folder_path):
            # 检查文件名是否以指定前缀开头
            if filename.startswith(prefix):
                # 可以选择性地添加文件的完整路径
                full_path = os.path.join(folder_path, filename)
                matching_files.append(full_path)
    except FileNotFoundError:
        print(f"错误：文件夹 '{folder_path}' 未找到。")
    return matching_files