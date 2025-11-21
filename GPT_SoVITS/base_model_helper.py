import os
import torch
from config import (device, is_half)
from GPT_SoVITS.module.models import Generator

now_dir = os.getcwd()
# 初始化
hifigan_model = None
bigvgan_model = None
sv_cn_model = None

def clean_hifigan_model():
    global hifigan_model
    if hifigan_model:
        hifigan_model = hifigan_model.cpu()
        hifigan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass

def clean_bigvgan_model():
    global bigvgan_model
    if bigvgan_model:
        bigvgan_model = bigvgan_model.cpu()
        bigvgan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def clean_sv_cn_model():
    global sv_cn_model
    if sv_cn_model:
        sv_cn_model.embedding_model = sv_cn_model.embedding_model.cpu()
        sv_cn_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def init_bigvgan():
    global bigvgan_model, hifigan_model, sv_cn_model
    from BigVGAN import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False,
    )  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    clean_hifigan_model()
    clean_sv_cn_model()
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)


def init_hifigan():
    global hifigan_model, bigvgan_model, sv_cn_model
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0,
        is_bias=True,
    )
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load(
        "%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,),
        map_location="cpu",
        weights_only=False,
    )
    print("loading vocoder", hifigan_model.load_state_dict(state_dict_g))
    clean_bigvgan_model()
    clean_sv_cn_model()
    if is_half == True:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)


def init_sv_cn():
    global hifigan_model, bigvgan_model, sv_cn_model
    from sv import SV
    sv_cn_model = SV(device, is_half)
    clean_bigvgan_model()
    clean_hifigan_model()


def init_model_by_version(version: str):
    if version == "v3":
        base_model = init_bigvgan()
        print("初始化BigVGAN模型...")
    elif version == "v4":
        print("初始化HiFi-GAN模型...")
        base_model = init_hifigan()  
    elif version in ["v2Pro", "v2ProPlus"]:
        print("初始化中文说话人编码模型...")
        base_model = init_sv_cn()

def clear_all_models():
    clean_bigvgan_model()
    clean_hifigan_model()
    clean_sv_cn_model()
    

def get_all_models():
    global hifigan_model, bigvgan_model, sv_cn_model
    return hifigan_model, bigvgan_model, sv_cn_model