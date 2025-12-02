import torch
import os
import torchaudio
from api_interface.config import (is_half,sv_path)
from api_interface.dataset.common import (get_device,save_pth,get_sv_cn_dir,get_wav32dir)
from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi


maxx = 0.95
alpha = 0.5


class SV:
    def __init__(self, device:str,sv_path:str):
        pretrained_state = torch.load(sv_path, map_location="cpu")
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        self.res = torchaudio.transforms.Resample(32000, 16000).to(device)
        if is_half == False:
            self.embedding_model = self.embedding_model.to(device)
        else:
            self.embedding_model = self.embedding_model.half().to(device)
        self.is_half = is_half

    def compute_embedding3(self, wav):  # (1,x)#-1~1
        with torch.no_grad():
            wav = self.res(wav)
            if self.is_half == True:
                wav = wav.half()
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            )
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb

def name2go(wav_name:str,
            device:str,
            sv:SV,
            sv_cn_dir:str,
            wav32dir:str):
    """
    处理单个wav文件
    参数:
    wav_name: wav文件名
    device: 设备
    sv: sv模型
    sv_cn_dir: sv_cn文件夹
    wav32dir: wav32k文件夹
    """
    sv_cn_path = "%s/%s.pt" % (sv_cn_dir, wav_name)
    if os.path.exists(sv_cn_path):
        return
    wav_path = "%s/%s" % (wav32dir, wav_name)
    wav32k, sr0 = torchaudio.load(wav_path)
    assert sr0 == 32000
    wav32k = wav32k.to(device)
    emb = sv.compute_embedding3(wav32k).cpu()  # torch.Size([1, 20480])
    save_pth(emb, sv_cn_path)



def train_sv(opt_dir:str,
              train_dataset_list:list[dict[str, str]],
              )->None:
    """
    训练sv
    参数:
    opt_dir: 数据集路径
    train_dataset_list: 训练数据集列表
    """
    device = get_device()
    sv = SV(device=device,sv_path=sv_path)
    sv_cn_dir = get_sv_cn_dir(opt_dir=opt_dir)
    wav32dir = get_wav32dir(opt_dir=opt_dir)
    for dataset in train_dataset_list:
        try:
            text = dataset.get("text")
            wav_path = dataset.get("wav_path")
            wav_name = os.path.basename(wav_path)
            name2go(wav_name=wav_name,
                    device=device,
                    sv=sv,
                    sv_cn_dir=sv_cn_dir,
                    wav32dir=wav32dir)
        except:
            print(f"{dataset}加入训练任务失败")
