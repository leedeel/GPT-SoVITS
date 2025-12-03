import os
import json
from api_interface.config import (
    exp_root,
    python_exec,
    SoVITS_weight_version2root,
    is_half,
    pretrained_sovits_name,
    GPU_INDEX
)
from tools.my_utils import check_details, check_for_existance
from subprocess import Popen
from api_interface.model_training.common import get_tmp_dir


gpus = "-".join(map(str, GPU_INDEX))
p_train_SoVITS = None

def check_sovits_train_status():
    global p_train_SoVITS
    if p_train_SoVITS != None:
        return {"code":-1, "msg": "SoVITS模型正在训练中"}
    return {"code":0, "msg": "SoVITS模型未在训练中"}

def train_sovits(
    version:str,
    exp_name:str,
    lora_rank:int=32,
    text_low_lr_rate:float=0.4,
    batch_size:int=4,
    total_epoch:int=8,
    save_every_epoch:int=4,
    if_save_latest:bool=True,
    if_save_every_weights:bool=True,
    if_grad_ckpt:bool=False,
):
    """
    训练SoVITS模型
    :param exp_name: 实验名称
    :param version: 模型版本
    :param lora_rank: lora rank
    :param text_low_lr_rate: 文本低学习率
    :param batch_size: 批处理大小
    :param total_epoch: 总epoch数
    :param save_every_epoch: 每隔多少epoch保存一次模型
    :param if_save_latest: 是否保存最新模型
    :param if_save_every_weights: 是否保存每一次的模型
    :param if_grad_ckpt: 是否使用梯度检查点
    :return: None
    """
    global p_train_SoVITS
    if p_train_SoVITS != None:
        return {"code":-1, "msg": "SoVITS模型正在训练中"}
    
    exp_name = exp_name.rstrip(" ")
    config_file = (
        "GPT_SoVITS/configs/s2.json"
        if version not in {"v2Pro", "v2ProPlus"}
        else f"GPT_SoVITS/configs/s2{version}.json"
    )
    with open(config_file) as f:
        data = f.read()
        data = json.loads(data)
    s2_dir = "%s/%s" % (exp_root, exp_name)
    os.makedirs("%s/logs_s2_%s" % (s2_dir, version), exist_ok=True)
    if check_for_existance([s2_dir], is_train=True):
        check_details([s2_dir], is_train=True)
    if is_half == False:
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2)
    
    pretrained_s2G = pretrained_sovits_name[version]
    pretrained_s2D = pretrained_sovits_name[version].replace("s2G", "s2D")
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = text_low_lr_rate
    data["train"]["pretrained_s2G"] = pretrained_s2G
    data["train"]["pretrained_s2D"] = pretrained_s2D
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["save_every_epoch"] = save_every_epoch
    data["train"]["gpu_numbers"] = gpus
    data["train"]["grad_ckpt"] = if_grad_ckpt
    data["train"]["lora_rank"] = lora_rank
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
    data["save_weight_dir"] = SoVITS_weight_version2root[version]
    data["name"] = exp_name
    data["version"] = version
    tmp_dir = get_tmp_dir()
    tmp_config_path = "%s/tmp_s2.json" % tmp_dir
    with open(tmp_config_path, "w") as f:
        f.write(json.dumps(data))
    if version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
        cmd = '"%s" -s GPT_SoVITS/s2_train.py --config "%s"' % (python_exec, tmp_config_path)
    else:
        cmd = '"%s" -s GPT_SoVITS/s2_train_v3_lora.py --config "%s"' % (python_exec, tmp_config_path)
    print(cmd)
    p_train_SoVITS = Popen(cmd, shell=True)
    p_train_SoVITS.wait()
    p_train_SoVITS = None
    return {"code":0, "msg": "SoVITS模型训练完成"}
