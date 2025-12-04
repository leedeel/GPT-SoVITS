import os
import gc
import torch
import numpy as np
from api_interface.config import (exp_root)
from api_interface.dataset.common import (check_memory_usage)

def train_dataset(
    version: str,
    train_dataset_list: list[dict],
    exp_name: str,
    language:str = "zh"
):
    """
    数据集训练 - 修复multiprocessing CUDA错误版本
    """
    # 设置multiprocessing启动方式
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    opt_dir = os.path.join(exp_root, exp_name)
    os.makedirs(opt_dir, exist_ok=True)
    print(f"待处理数据集:{train_dataset_list}，版本:{version}，语言:{language}，输出路径:{opt_dir}")
    
    def cleanup_memory():
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 创建上下文管理器
    ctx = mp.get_context('spawn')
    
    def run_tfe_process():
        """在独立进程中运行文本分词特征提取"""
        print("1.文本分词特征提取")
        check_memory_usage()
        from api_interface.dataset.tfe_fn import train_tfe
        train_tfe(version=version,
                  opt_dir=opt_dir,
                  train_dataset_list=train_dataset_list,
                  language=language)
    
    def run_ssl_process():
        """在独立进程中运行语音自监督特征提取"""
        print("2.语音自监督特征提取")
        check_memory_usage()
        from api_interface.dataset.ssl_fn import train_ssl
        train_ssl(opt_dir=opt_dir,
                  train_dataset_list=train_dataset_list)
    
    def run_sv_process():
        """在独立进程中运行声纹训练"""
        print("3.声纹训练")
        check_memory_usage()
        from api_interface.dataset.sv_fn import train_sv
        train_sv(opt_dir=opt_dir,
                 train_dataset_list=train_dataset_list)
    
    def run_semantic_token_process():
        """在独立进程中运行语义token提取"""
        print("4.语义token提取")
        check_memory_usage()
        from api_interface.dataset.semantic_token_fn import train_semantic_token
        train_semantic_token(opt_dir=opt_dir,
                             version=version,
                             train_dataset_list=train_dataset_list)
    
    processes = []
    
    try:
        # 1.文本分词特征提取
        print("1.文本分词特征提取")
        p1 = ctx.Process(target=run_tfe_process)
        p1.start()
        p1.join()
        if p1.exitcode != 0:
            raise RuntimeError("文本分词特征提取失败")
        cleanup_memory()
        
        # 2.语音自监督特征提取
        print("2.语音自监督特征提取")
        p2 = ctx.Process(target=run_ssl_process)
        p2.start()
        p2.join()
        if p2.exitcode != 0:
            raise RuntimeError("语音自监督特征提取失败")
        cleanup_memory()
        
        if "Pro" in version:
            # 3.声纹训练
            print("3.声纹训练")
            p3 = ctx.Process(target=run_sv_process)
            p3.start()
            p3.join()
            if p3.exitcode != 0:
                raise RuntimeError("声纹训练失败")
            cleanup_memory()
        
        # 4.语义token提取
        print("4.语义token提取")
        p4 = ctx.Process(target=run_semantic_token_process)
        p4.start()
        p4.join()
        if p4.exitcode != 0:
            raise RuntimeError("语义token提取失败")
        cleanup_memory()
        
    finally:
        # 清理进程
        for p in [p1, p2, p3, p4] if "Pro" in version else [p1, p2, p4]:
            if hasattr(p, 'close'):
                p.close()
        
        print("所有训练步骤完成")