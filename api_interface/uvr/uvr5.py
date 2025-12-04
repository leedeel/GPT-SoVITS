import logging
import os
import traceback
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

logger = logging.getLogger(__name__)
import ffmpeg
import torch
from tools.uvr5.bsroformer import Roformer_Loader
from tools.uvr5.mdxnet import MDXNetDereverb
from tools.uvr5.vr import AudioPre, AudioPreDeEcho
from api_interface.config import (is_half,exp_root)
from api_interface.uvr.common import (get_device,get_urv5_file_name_opt_dir,get_tmp_dir,find_files_with_prefix)

WEIGHT_URV5_ROOT = "tools/uvr5/uvr5_weights"




def get_uvr5_names():
    
    uvr5_names = []
    for name in os.listdir(WEIGHT_URV5_ROOT):
        if name.endswith(".pth") or name.endswith(".ckpt") or "onnx" in name:
            uvr5_names.append(name.replace(".pth", "").replace(".ckpt", ""))
    return uvr5_names


def init_uvr5_model(model_name,
                    device,
                    agg,
                    ):
    """
    初始化uvr5模型
    :param model_name: 模型名称
    :param device: 设备
    :param agg: 聚合度
    :return: 模型对象
    """
    if model_name == "onnx_dereverb_By_FoxJoy":
        pre_fun = MDXNetDereverb(15)
    elif "roformer" in model_name.lower():
        func = Roformer_Loader
        pre_fun = func(
            model_path=os.path.join(WEIGHT_URV5_ROOT, model_name + ".ckpt"),
            config_path=os.path.join(WEIGHT_URV5_ROOT, model_name + ".yaml"),
            device=device,
            is_half=is_half,
        )
        if not os.path.exists(os.path.join(WEIGHT_URV5_ROOT, model_name + ".yaml")):
            raise Exception(f"配置文件不存在:{os.path.join(WEIGHT_URV5_ROOT, model_name + '.yaml')}")
    else:
        func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
        pre_fun = func(
            agg=int(agg),
            model_path=os.path.join(WEIGHT_URV5_ROOT, model_name + ".pth"),
            device=device,
            is_half=is_half,
        )
    return pre_fun


def process_uvr5(file_path,
                 file_name,
                 save_root_vocal,
                 save_root_ins,
                 pre_fun,
                 format0,
                 is_hp3):
    """
    处理uvr5
    :param inp_path: 输入路径
    :param save_root_vocal: 人声保存路径
    :param save_root_ins: 伴奏保存路径
    :param pre_fun: 模型对象
    :param format0: 格式
    :param is_hp3: 是否是hp3
    :return: 
    """
    try:
        info = ffmpeg.probe(file_path, cmd="ffprobe")
        if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":
            pre_fun._path_audio_(file_path, save_root_ins, save_root_vocal, format0, is_hp3)
        else:
            tmp_path = "%s/%s.reformatted.wav" % (get_tmp_dir(),file_name)
            os.system(f'ffmpeg -i "{file_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
            pre_fun._path_audio_(tmp_path, save_root_ins, save_root_vocal, format0, is_hp3)
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    except:
        traceback.print_exc()
    



def uvr(exp_name,
        model_name, 
        paths:list[str], 
        agg:int, 
        format0:str):
    """
    人声分离
    :param exp_name: 实验名称
    :param model_name: 模型名称
    :param paths: 文件路径列表
    :param agg: 聚合度 0-100
    :param format0: 格式, wav,flac,mp3,m4a
    :return: 
    """
    try:
        device = get_device()
        is_hp3 = "HP3" in model_name
        pre_fun = init_uvr5_model(model_name=model_name,device=device,agg=agg)
        VOCAL_PREFIX = "vocal_"
        INS_PREFIX = "instrument_"
        
        infos = []
        for file_path in paths:
            file_name = os.path.basename(file_path)
            urv5_file_name_opt_dir = get_urv5_file_name_opt_dir(root_dir=exp_root,exp_name=exp_name,file_name=file_name)
            # 在vrv5_file_name_opt_dir中查找vocal.wav和ins.wav
            vocal_matchs = find_files_with_prefix(folder_path=urv5_file_name_opt_dir, prefix=VOCAL_PREFIX)
            ins_matchs = find_files_with_prefix(folder_path=urv5_file_name_opt_dir, prefix=INS_PREFIX)
            if len(vocal_matchs) > 0 and len(ins_matchs) > 0:
                infos.append({
                    "name": file_name,
                    "path": file_path,
                    "vocal_path": vocal_matchs[0],
                    "ins_path": ins_matchs[0],
                    "status": "success",
                    "message": "文件已存在"
                })
                continue
            
            if os.path.isfile(file_path) == False:
                infos.append({
                    "name": file_name,
                    "path": file_path,
                    "status": "error",
                    "message": "文件不存在"
                })
                continue
            print(f"开始处理文件:{file_path}")
            process_uvr5(file_path=file_path,
                         file_name=file_name,
                         save_root_vocal=urv5_file_name_opt_dir,
                         save_root_ins=urv5_file_name_opt_dir,
                         pre_fun=pre_fun,
                         format0=format0,
                         is_hp3=is_hp3)
            print(f"处理文件:{file_path}完成")
            vocal_matchs = find_files_with_prefix(folder_path=urv5_file_name_opt_dir, prefix=VOCAL_PREFIX)
            ins_matchs = find_files_with_prefix(folder_path=urv5_file_name_opt_dir, prefix=INS_PREFIX)
            if len(vocal_matchs) > 0 and len(ins_matchs) > 0:
                infos.append({
                    "name": file_name,
                    "path": file_path,
                    "vocal_path": vocal_matchs[0],
                    "ins_path": ins_matchs[0],
                    "status": "success",
                    "message": "文件已生成"
                })
            else:
                infos.append({
                    "name": file_name,
                    "path": file_path,
                    "status": "error",
                    "message": "文件生成失败"
                })
            
        return infos
    except Exception as e:
        print(f"人声分离时报错:{e}")
        raise e
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
