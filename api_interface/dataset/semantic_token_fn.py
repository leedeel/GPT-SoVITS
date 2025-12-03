import os
import torch
import GPT_SoVITS.utils as utils
from api_interface.config import (is_half,pretrained_sovits_name)
from api_interface.dataset.common import (get_device,get_hubert_dir,get_semantic_path)



def pretrained_version_by_pretrained_s2G(pretrained_s2G:str):
    """
    根据预训练模型路径获取预训练版本
    参数:
    pretrained_s2G: 预训练模型路径
    返回:
    预训练版本
    """
    size = os.path.getsize(pretrained_s2G)
    if size < 82978 * 1024:
        version = "v1"
    elif size < 100 * 1024 * 1024:
        version = "v2"
    elif size < 103520 * 1024:
        version = "v1"
    elif size < 700 * 1024 * 1024:
        version = "v2"
    else:
        version = "v3"
    return version


def train_semantic_token(opt_dir:str,
                         version:str,
                         train_dataset_list:list[dict[str, str]]):
    """
    语义token提取
    参数:
    opt_dir: 数据集路径
    train_dataset_list: 训练数据集列表
    """
    semantic_path = get_semantic_path(opt_dir=opt_dir)
    if os.path.exists(semantic_path):
        print(f"语义token已存在:{semantic_path}")
        return
    
    if version not in pretrained_sovits_name.keys():
        raise ValueError(f"版本{version}不支持语义token提取")
    pretrained_s2G = pretrained_sovits_name[version]
    print(f"预训练模型路径:{pretrained_s2G}")
    if not os.path.exists(pretrained_s2G):
        raise FileNotFoundError(pretrained_s2G)
    # 获取预训练版本
    pretrained_version = pretrained_version_by_pretrained_s2G(pretrained_s2G=pretrained_s2G)
    print(f"预训练版本:{pretrained_version}")
    # 获取设备
    device = get_device()
    s2config_path = ( "GPT_SoVITS/configs/s2.json"
                    if version not in {"v2Pro", "v2ProPlus"}
                    else f"GPT_SoVITS/configs/s2{version}.json")
    print(f"s2config_path:{s2config_path}")
    hps = utils.get_hparams_from_file(s2config_path)
    if pretrained_version != "v3":
        from GPT_SoVITS.module.models import SynthesizerTrn
    else:
        from GPT_SoVITS.module.models import SynthesizerTrnV3 as SynthesizerTrn
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version=version,
        **hps.model,
    )
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    vq_model.load_state_dict(
            torch.load(pretrained_s2G, map_location="cpu", weights_only=False)["weight"], strict=False
        )
    hubert_dir = get_hubert_dir(opt_dir=opt_dir)
    
    result_list = []
    for dataset in train_dataset_list:
        try:
            text = dataset.get("text")
            wav_path = dataset.get("wav_path")
            wav_name = os.path.basename(wav_path)
            semantic = name2go(wav_name=wav_name,
                               hubert_dir=hubert_dir,
                               device=device,
                               vq_model=vq_model)
            result_list.append(semantic)
        except Exception as e:
            print(f"语义token提取失败:{dataset},{e}")
    
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(result_list))
    return semantic_path
        

def name2go(wav_name:str,hubert_dir:str,device:str,vq_model:any):
    hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
    if os.path.exists(hubert_path) == False:
        return
    ssl_content = torch.load(hubert_path, map_location="cpu")
    if is_half == True:
        ssl_content = ssl_content.half().to(device)
    else:
        ssl_content = ssl_content.to(device)
    codes = vq_model.extract_latent(ssl_content)
    semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
    return "%s\t%s" % (wav_name, semantic)