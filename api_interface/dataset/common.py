import os
from time import time as ttime
import shutil
import torch

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

def save_pth(fea, path):  #####fix issue: torch.save doesn't support chinese path
    """
    保存pth文件，解决中文路径问题
    参数:
    fea: 特征
    path: 路径
    """
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % (ttime())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))

def get_tfe_file_path(opt_dir:str):
    """
    获取tfe文件路径
    参数:
    opt_dir: 存储路径
    返回:
    tfe文件路径
    """
    tfe_file_path = os.path.join(opt_dir, "2-name2text.txt")
    return tfe_file_path


def get_bert_dir(opt_dir:str):
    """
    获取bert特征存储路径
    参数:
    opt_dir: 存储路径
    返回:
    bert特征存储路径
    """
    bert_dir = os.path.join(opt_dir, "3-bert")
    os.makedirs(bert_dir, exist_ok=True)
    return bert_dir

def get_hubert_dir(opt_dir:str):
    """
    获取hubert特征存储路径
    参数:
    opt_dir: 存储路径
    返回:
    hubert特征存储路径
    """
    hubert_dir = os.path.join(opt_dir, "4-cnhubert")
    os.makedirs(hubert_dir, exist_ok=True)
    return hubert_dir

def get_wav32dir(opt_dir:str):
    """
    获取wav32k特征存储路径
    参数:
    opt_dir: 存储路径
    返回:
    wav32k特征存储路径
    """
    wav32dir = os.path.join(opt_dir, "5-wav32k")
    os.makedirs(wav32dir, exist_ok=True)
    return wav32dir

def get_sv_cn_dir(opt_dir:str):
    """
    获取sv_cn特征存储路径
    参数:
    opt_dir: 存储路径
    返回:
    sv_cn特征存储路径
    """
    sv_cn_dir = os.path.join(opt_dir, "7-sv_cn")
    os.makedirs(sv_cn_dir, exist_ok=True)
    return sv_cn_dir


def get_semantic_path(opt_dir:str):
    """
    获取语义token存储路径
    参数:
    opt_dir: 存储路径
    返回:
    语义token存储路径
    """
    semantic_path = os.path.join(opt_dir, "6-name2semantic.tsv")
    return semantic_path

def check_memory_usage():
    """检查内存使用情况"""
    import torch
    
    print("=== 内存使用情况 ===")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU: {allocated:.2f} GB 已分配, {cached:.2f} GB 预留")
    
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RAM: {mem_info.rss / 1024**3:.2f} GB")