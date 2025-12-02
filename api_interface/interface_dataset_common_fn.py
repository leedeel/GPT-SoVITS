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