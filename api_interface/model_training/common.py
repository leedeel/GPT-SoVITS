import os

def get_tmp_dir():
    now_dir = os.getcwd()
    tmp = os.path.join(now_dir, "TEMP","model_training")
    os.makedirs(tmp, exist_ok=True)
    return tmp


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