
import os
from api_interface.config import (exp_root)
from api_interface.dataset.tfe_fn import train_tfe
from api_interface.dataset.ssl_fn import train_ssl
from api_interface.dataset.sv_fn import train_sv
from api_interface.dataset.semantic_token_fn import train_semantic_token
from api_interface.dataset.common import check_memory_usage
import gc


def cleanup_memory():
    """强制清理内存"""
    import torch
    gc.collect()  # 垃圾回收
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理GPU缓存
        torch.cuda.reset_max_memory_allocated()  # 重置内存统计
    try:
        torch.cuda.ipc_collect()  # 收集共享内存
    except:
        pass


def train_dataset(
    version: str,
    train_dataset_list: list[dict],
    exp_name: str,
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
    opt_dir = os.path.join(exp_root, exp_name)
    os.makedirs(opt_dir, exist_ok=True)
    print(f"待处理数据集:{train_dataset_list}，版本:{version}，语言:{language}，输出路径:{opt_dir}")
    try:
        # 1.文本分词特征提取
        print(f"1.文本分词特征提取")
        check_memory_usage()
        train_tfe(version=version,
                opt_dir=opt_dir,
                train_dataset_list=train_dataset_list,
                language=language)
        cleanup_memory()
        print(f"1.文本分词特征提取完成")
        
        
        # 2.语音自监督特征提取
        print(f"2.语音自监督特征提取")
        check_memory_usage()
        train_ssl(opt_dir=opt_dir,
                train_dataset_list=train_dataset_list)
        cleanup_memory()
        print(f"2.语音自监督特征提取完成")
        
        if "Pro" in version:
            # 3.声纹训练
            print(f"3.声纹训练")
            check_memory_usage()
            train_sv(opt_dir=opt_dir,
                    train_dataset_list=train_dataset_list)
            cleanup_memory()
            print(f"3.声纹训练完成")
            check_memory_usage()
        
        # 4.语义token提取
        print(f"4.语义token提取")
        check_memory_usage()
        train_semantic_token(opt_dir=opt_dir,
                            version=version,
                            train_dataset_list=train_dataset_list)
        cleanup_memory()
        print(f"4.语义token提取完成")
        check_memory_usage()
    except Exception as e:
        print(f"数据集训练失败: {e}")
        raise e
    finally:
        cleanup_memory()
    
    
    
    
    
            
        
        
        
    
    
    