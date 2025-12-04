import os
import multiprocessing as mp
from api_interface.config import (exp_root)

def run_tfe(version, opt_dir, train_dataset_list, language):
    """在独立进程中运行文本分词特征提取"""
    from api_interface.dataset.tfe_fn import train_tfe
    return train_tfe(version=version,
                     opt_dir=opt_dir,
                     train_dataset_list=train_dataset_list,
                     language=language)

def run_ssl(opt_dir, train_dataset_list):
    """在独立进程中运行语音自监督特征提取"""
    from api_interface.dataset.ssl_fn import train_ssl
    return train_ssl(opt_dir=opt_dir,
                     train_dataset_list=train_dataset_list)

def run_sv(opt_dir, train_dataset_list):
    """在独立进程中运行声纹训练"""
    from api_interface.dataset.sv_fn import train_sv
    return train_sv(opt_dir=opt_dir,
                    train_dataset_list=train_dataset_list)

def run_semantic_token(opt_dir, version, train_dataset_list):
    """在独立进程中运行语义token提取"""
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
    使用进程隔离的训练函数 - 最彻底的内存清理
    """
    opt_dir = os.path.join(exp_root, exp_name)
    os.makedirs(opt_dir, exist_ok=True)
    
    processes = []
    results = []
    
    try:
        # 步骤1：文本分词特征提取
        print("步骤1：文本分词特征提取")
        p1 = mp.Process(target=run_tfe,
                        args=(version, opt_dir, train_dataset_list, language))
        p1.start()
        p1.join()
        if p1.exitcode != 0:
            raise RuntimeError("文本分词特征提取失败")
        processes.append(p1)
        print("步骤1完成")
        
        # 步骤2：语音自监督特征提取
        print("步骤2：语音自监督特征提取")
        p2 = mp.Process(target=run_ssl,
                        args=(opt_dir, train_dataset_list))
        p2.start()
        p2.join()
        if p2.exitcode != 0:
            raise RuntimeError("语音自监督特征提取失败")
        processes.append(p2)
        print("步骤2完成")
        
        if "Pro" in version:
            # 步骤3：声纹训练
            print("步骤3：声纹训练")
            p3 = mp.Process(target=run_sv,
                            args=(opt_dir, train_dataset_list))
            p3.start()
            p3.join()
            if p3.exitcode != 0:
                raise RuntimeError("声纹训练失败")
            processes.append(p3)
            print("步骤3完成")
        
        # 步骤4：语义token提取
        print("步骤4：语义token提取")
        p4 = mp.Process(target=run_semantic_token,
                        args=(opt_dir, version, train_dataset_list))
        p4.start()
        p4.join()
        if p4.exitcode != 0:
            raise RuntimeError("语义token提取失败")
        processes.append(p4)
        print("步骤4完成")
        
    finally:
        # 确保所有进程终止
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        
        # 清理进程资源
        for p in processes:
            p.close()
        print("所有训练步骤完成，进程已清理")