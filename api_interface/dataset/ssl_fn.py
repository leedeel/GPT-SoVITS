import os
import torch
from api_interface.config import (is_half,cnhubert_path)
from GPT_SoVITS.feature_extractor import cnhubert
import numpy as np
from scipy.io import wavfile
import librosa
from tools.my_utils import load_audio
from api_interface.dataset.common import (get_device, save_pth, get_hubert_dir, get_wav32dir)

def init_cnhubert_model(ssl_pretrained_dir:str,
                        device:str)->any:
    """
    初始化cnhubert模型
    参数:
    ssl_pretrained_dir: cnhubert预训练模型路径
    device: 设备
    返回:
    cnhubert模型
    """
    cnhubert.cnhubert_base_path = ssl_pretrained_dir
    model = cnhubert.get_model()
    if is_half == True:
        model = model.half().to(device)
    else:
        model = model.to(device)
    return model


maxx = 0.95
alpha = 0.5

def name2go(wav_name:str, 
            wav_path:str, 
            nan_fails:list[str],
            hubert_dir:str,
            wav32dir:str,
            model:any, 
            device:str):
    
    hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
    if os.path.exists(hubert_path):
        return
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (wav_name, tmp_max))
        return
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha * 32768)) + ((1 - alpha) * 32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha * 1145.14)) + ((1 - alpha) * 1145.14) * tmp_audio
    tmp_audio = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)  # 不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if is_half == True:
        tensor_wav16 = tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()  # torch.Size([1, 768, 215])
    if np.isnan(ssl.detach().numpy()).sum() != 0:
        nan_fails.append((wav_name, wav_path))
        print("nan filtered:%s" % wav_name)
        return
    wavfile.write(
        "%s/%s" % (wav32dir, wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )
    save_pth(ssl, hubert_path)



def train_ssl(opt_dir:str,
              train_dataset_list:list[dict[str, str]])->None:
    device = get_device()
    model = init_cnhubert_model(ssl_pretrained_dir=cnhubert_path,
                                device=device)
    nan_fails = []
    hubert_dir = get_hubert_dir(opt_dir=opt_dir)
    wav32dir = get_wav32dir(opt_dir=opt_dir)
    for dataset in train_dataset_list:
        try:
            text = dataset.get("text")
            wav_path = dataset.get("wav_path")
            wav_name = os.path.basename(wav_path)
            name2go(wav_name=wav_name,
                    wav_path=wav_path,
                    hubert_dir=hubert_dir,
                    wav32dir=wav32dir,
                    nan_fails=nan_fails,
                    model=model,
                    device=device)
        except:
            print(f"{dataset}加入训练任务失败")
            
    if len(nan_fails) > 0 and is_half == True:
        is_half = False
        model = model.float()
        for wav in nan_fails:
            try:
                name2go(wav_name=wav[0], 
                        wav_path=wav[1],
                        hubert_dir=hubert_dir,
                        wav32dir=wav32dir,
                        nan_fails=nan_fails,
                        model=model,
                        device=device)
            except:
                print(wav_name)
