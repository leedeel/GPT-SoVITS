import os
import gc
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from api_interface.config import (exp_root)
from api_interface.dataset.common import (check_memory_usage)

# 定义模块级别的任务函数（不能在train_dataset内部定义）
def _task_tfe(version, opt_dir, train_dataset_list, language):
    """模块级别的文本分词特征提取任务"""
    from api_interface.dataset.tfe_fn import train_tfe
    return train_tfe(version=version,
                     opt_dir=opt_dir,
                     train_dataset_list=train_dataset_list,
                     language=language)

def _task_ssl(opt_dir, train_dataset_list):
    """模块级别的语音自监督特征提取任务"""
    from api_interface.dataset.ssl_fn import train_ssl
    return train_ssl(opt_dir=opt_dir,
                     train_dataset_list=train_dataset_list)

def _task_sv(opt_dir, train_dataset_list):
    """模块级别的声纹训练任务"""
    from api_interface.dataset.sv_fn import train_sv
    return train_sv(opt_dir=opt_dir,
                    train_dataset_list=train_dataset_list)

def _task_semantic_token(opt_dir, version, train_dataset_list):
    """模块级别的语义token提取任务"""
    from api_interface.dataset.semantic_token_fn import train_semantic_token
    return train_semantic_token(opt_dir=opt_dir,
                                version=version,
                                train_dataset_list=train_dataset_list)

def train_dataset(
    version: str,
    train_dataset_list: list[dict],
    exp_name: str,
    language:str = "zh"
):
    """
    数据集训练 - 使用ProcessPoolExecutor版本，修复pickle错误
    """
    # 设置multiprocessing启动方式为spawn
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过，忽略
    
    opt_dir = os.path.join(exp_root, exp_name)
    os.makedirs(opt_dir, exist_ok=True)
    
    def cleanup_memory():
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 使用ProcessPoolExecutor，指定spawn上下文
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        try:
            # 1.文本分词特征提取
            print("1.文本分词特征提取")
            check_memory_usage()
            future1 = executor.submit(_task_tfe, version, opt_dir, train_dataset_list, language)
            result1 = future1.result()  # 等待完成
            print("1.文本分词特征提取完成")
            cleanup_memory()
            
            # 2.语音自监督特征提取
            print("2.语音自监督特征提取")
            check_memory_usage()
            future2 = executor.submit(_task_ssl, opt_dir, train_dataset_list)
            result2 = future2.result()
            print("2.语音自监督特征提取完成")
            cleanup_memory()
            
            if "Pro" in version:
                # 3.声纹训练
                print("3.声纹训练")
                check_memory_usage()
                future3 = executor.submit(_task_sv, opt_dir, train_dataset_list)
                result3 = future3.result()
                print("3.声纹训练完成")
                cleanup_memory()
            
            # 4.语义token提取
            print("4.语义token提取")
            check_memory_usage()
            future4 = executor.submit(_task_semantic_token, opt_dir, version, train_dataset_list)
            result4 = future4.result()
            print("4.语义token提取完成")
            cleanup_memory()
            check_memory_usage()
        except Exception as e:
            print(f"训练出错: {e}")
            raise
    
    print("所有训练步骤完成")