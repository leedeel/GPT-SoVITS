import os
import gc
import torch
from concurrent.futures import ProcessPoolExecutor
from api_interface.config import (exp_root)
from api_interface.dataset.common import (check_memory_usage)

def train_dataset(
    version: str,
    train_dataset_list: list[dict],
    exp_name: str,
    language:str = "zh"
):
    """
    数据集训练 - 使用ProcessPoolExecutor版本
    """
    opt_dir = os.path.join(exp_root, exp_name)
    os.makedirs(opt_dir, exist_ok=True)
    
    # 定义任务函数
    def task_tfe():
        from api_interface.dataset.tfe_fn import train_tfe
        return train_tfe(version=version,
                         opt_dir=opt_dir,
                         train_dataset_list=train_dataset_list,
                         language=language)
    
    def task_ssl():
        from api_interface.dataset.ssl_fn import train_ssl
        return train_ssl(opt_dir=opt_dir,
                         train_dataset_list=train_dataset_list)
    
    def task_sv():
        from api_interface.dataset.sv_fn import train_sv
        return train_sv(opt_dir=opt_dir,
                        train_dataset_list=train_dataset_list)
    
    def task_semantic_token():
        from api_interface.dataset.semantic_token_fn import train_semantic_token
        return train_semantic_token(opt_dir=opt_dir,
                                    version=version,
                                    train_dataset_list=train_dataset_list)
    
    # 使用ProcessPoolExecutor，它会自动使用spawn方式
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as executor:
        try:
            # 1.文本分词特征提取
            print("1.文本分词特征提取")
            check_memory_usage()
            future1 = executor.submit(task_tfe)
            result1 = future1.result()  # 等待完成
            print("1.文本分词特征提取完成")
            
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 2.语音自监督特征提取
            print("2.语音自监督特征提取")
            check_memory_usage()
            future2 = executor.submit(task_ssl)
            result2 = future2.result()
            print("2.语音自监督特征提取完成")
            
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if "Pro" in version:
                # 3.声纹训练
                print("3.声纹训练")
                check_memory_usage()
                future3 = executor.submit(task_sv)
                result3 = future3.result()
                print("3.声纹训练完成")
                
                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 4.语义token提取
            print("4.语义token提取")
            check_memory_usage()
            future4 = executor.submit(task_semantic_token)
            result4 = future4.result()
            print("4.语义token提取完成")
            
        except Exception as e:
            print(f"训练出错: {e}")
            raise
    
    print("所有训练步骤完成")