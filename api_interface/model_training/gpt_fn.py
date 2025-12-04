from api_interface.config import (
    infer_device,
    exp_root,
    python_exec,
    GPT_weight_version2root,
    is_half,
    pretrained_gpt_name,
    GPU_INDEX
)
from tools.my_utils import check_details, check_for_existance
from subprocess import Popen
import yaml
import os
import gc
from api_interface.model_training.common import get_tmp_dir,check_memory_usage

gpus = "-".join(map(str, GPU_INDEX))
default_gpu_numbers = infer_device.index
set_gpu_numbers = GPU_INDEX

p_train_GPT = None

def check_gpt_train_status():
    global p_train_GPT
    if p_train_GPT != None:
        return {"code":-1, "msg": "GPT模型正在训练中"}
    return {"code":0, "msg": "GPT模型未在训练中"}


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input

def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs



def train_gpt(
    version:str,
    exp_name,
    batch_size:int=4,
    total_epoch:int=15,
    save_every_epoch:int=5,
    if_dpo:bool=False,
    if_save_latest:bool=True,
    if_save_every_weights:bool=True
):
    global p_train_GPT
    if p_train_GPT != None:
        return {"code":-1, "msg": "GPT模型正在训练中"}
    check_memory_usage()
    exp_name = exp_name.rstrip(" ")
    with open(
        "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml"
    ) as f:
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)
    s1_dir = "%s/%s" % (exp_root, exp_name)
    os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
    if check_for_existance([s1_dir], is_train=True):
        check_details([s1_dir], is_train=True)
    if is_half == False:
        data["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2)
    
    pretrained_s1 = pretrained_gpt_name[version]
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["pretrained_s1"] = pretrained_s1
    data["train"]["save_every_n_epoch"] = save_every_epoch
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_dpo"] = if_dpo
    data["train"]["half_weights_save_dir"] = GPT_weight_version2root[version]
    data["train"]["exp_name"] = exp_name
    data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
    data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
    data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)
    # data["version"]=version
    gpus = "-".join(map(str, GPU_INDEX))
    os.environ["_CUDA_VISIBLE_DEVICES"] = str(fix_gpu_numbers(gpus.replace("-", ",")))
    os.environ["hz"] = "25hz"
    tmp_dir = get_tmp_dir()
    tmp_config_path = "%s/tmp_s1.yaml" % tmp_dir
    with open(tmp_config_path, "w") as f:
        f.write(yaml.dump(data, default_flow_style=False))
    # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
    cmd = '"%s" -s GPT_SoVITS/s1_train.py --config_file "%s" ' % (python_exec, tmp_config_path)
    print(cmd)
    p_train_GPT = Popen(cmd, shell=True)
    p_train_GPT.wait()
    p_train_GPT = None
    cleanup_memory()
    check_memory_usage()
    return {"code":0, "msg": "GPT模型训练完成"}

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
