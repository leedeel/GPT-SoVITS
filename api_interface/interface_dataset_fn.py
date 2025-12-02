
import os
import sys
from transformers import AutoModelForMaskedLM, AutoTokenizer
from api_interface.config import (bert_path,cnhubert_path)
from tools import my_utils
from GPT_SoVITS.text.cleaner import clean_text
from time import time as ttime
import shutil
import torch
import json
from api_interface.interface_train_tfe_fn import train_tfe
from api_interface.interface_train_ssl_fn import train_ssl


def open1abc(
    version: str,
    train_dataset_list: list[dict],
    export_root: str,
    export_name: str,
    aviable_gpu_numbers: list[str],
    gpu_numbers1a,
    gpu_numbers1Ba,
    gpu_numbers1c,
    pretrained_s2G_path,
    language:str = "zh"
):
    """
    数据集训练
    参数:
    version: 版本号,可选值:["v1", "v2", "v4", "v2Pro", "v2ProPlus"]
    train_dataset_list: 训练数据集列表,每个元素为一个字典,包含以下字段:
    - text: 音频文本
    - wav_path: 音频路径
    
    """
    opt_dir = os.path.join(export_root, export_name)
    os.makedirs(opt_dir, exist_ok=True)
    
    # 1.文本分词特征提取
    tfe_process_file_path = train_tfe(version=version,
                                      opt_dir=opt_dir,
                                      train_dataset_list=train_dataset_list,
                                      language=language,
                                      bert_pretrained_dir=bert_path)
    
    # 2.语音自监督特征提取
    train_ssl(opt_dir=opt_dir,
              train_dataset_list=train_dataset_list,
              ssl_pretrained_dir=cnhubert_path)
    
    
    
    
            
        
        
        
    
    
    