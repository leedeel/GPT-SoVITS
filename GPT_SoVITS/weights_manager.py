import os
import torch
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from api_interface.config import  is_half, pretrained_sovits_name
from peft import LoraConfig, get_peft_model

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def change_sovits_weights(sovits_path:str):
    """
    更换SoVITS模型权重
    @param sovits_path: SoVITS模型路径
    @param device: 设备
    """
    # 获取SoviTS模型版本信息    
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    print(f"待加载SoVITS模型信息:sovits_path:{sovits_path}, symbol_version:{version}, model_version:{model_version}, if_lora_v3:{if_lora_v3}")

    # 检查LoRA权重对应的底模是否存在
    path_sovits_v3 = pretrained_sovits_name["v3"]
    path_sovits_v4 = pretrained_sovits_name["v4"]
    is_exist_s2gv3 = os.path.exists(path_sovits_v3)
    is_exist_s2gv4 = os.path.exists(path_sovits_v4)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    
    # 检查LoRA权重对应的底模是否存在
    if if_lora_v3 == True and is_exist == False:
        info = path_sovits + "SoVITS %s" % model_version + "底模缺失，无法加载相应 LoRA 权重"
        raise FileExistsError(info)
    
    # 加载SoVITS模型权重
    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"  # v3model,v2sybomls
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    if model_version not in {"v3", "v4"}:
        if "Pro" not in model_version:
            model_version = version
        else:
            hps.model.version = model_version
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    if if_lora_v3 == False:
        print("loading sovits_%s" % model_version, vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else:
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        print(
            "loading sovits_%spretrained_G" % model_version,
            vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False),
        )
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print("loading sovits_%s_lora%s" % (model_version, lora_rank))
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        # torch.save(vq_model.state_dict(),"merge_win.pth")
        vq_model.eval()

    yield (
        version, model_version, if_lora_v3, vq_model, hps
    )


def change_gpt_weights(gpt_path:str):
    """
    更换GPT模型权重
    @param gpt_path: GPT模型路径
    """
    print("更换GPT模型权重:", gpt_path)
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    dict_s1_config = dict_s1["config"]
    max_sec = dict_s1_config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(dict_s1_config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    # total = sum([param.nelement() for param in t2s_model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    yield (hz, max_sec, t2s_model)

