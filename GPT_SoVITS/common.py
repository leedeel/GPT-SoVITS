import torch
import torchaudio
from tools.i18n.i18n import I18nAuto, scan_language_list
from transformers import AutoModelForMaskedLM, AutoTokenizer
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from api_interface.config import  is_half

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
i18n=I18nAuto(language="zh_CN")
dict_language=None
bert_model=None
ssl_model=None
tokenizer=None

def init_device():
    """
    获取设备
    """
    global device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)
    return device

def init_dict_language(version:str):
    """
    初始化语言字典
    """
    global i18n,dict_language
    if i18n is None:
        i18n = I18nAuto(language="zh_CN")
    if dict_language is None:
        dict_language_v1 = {
            i18n("中文"): "all_zh",  # 全部按中文识别
            i18n("英文"): "en",  # 全部按英文识别#######不变
            i18n("日文"): "all_ja",  # 全部按日文识别
            i18n("中英混合"): "zh",  # 按中英混合识别####不变
            i18n("日英混合"): "ja",  # 按日英混合识别####不变
            i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
        }
        dict_language_v2 = {
            i18n("中文"): "all_zh",  # 全部按中文识别
            i18n("英文"): "en",  # 全部按英文识别#######不变
            i18n("日文"): "all_ja",  # 全部按日文识别
            i18n("粤语"): "all_yue",  # 全部按中文识别
            i18n("韩文"): "all_ko",  # 全部按韩文识别
            i18n("中英混合"): "zh",  # 按中英混合识别####不变
            i18n("日英混合"): "ja",  # 按日英混合识别####不变
            i18n("粤英混合"): "yue",  # 按粤英混合识别####不变
            i18n("韩英混合"): "ko",  # 按韩英混合识别####不变
            i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
            i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种
        }
        dict_language = dict_language_v1 if version == "v1" else dict_language_v2
        print("使用语言字典:", dict_language)
    return i18n,dict_language


def init_bert_model(bert_model_path:str):
    """
    初始化BERT模型
    bert_model_path: BERT模型路径
    device: 设备
    return:
    tokenizer: 分词器
    bert_model: BERT模型
    """
    global tokenizer,bert_model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    if bert_model is None:
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_model_path)
        device = init_device()
        if is_half == True:
            bert_model = bert_model.half().to(device)
        else:
            bert_model = bert_model.to(device)
    return tokenizer,bert_model

def init_ssl_model(ssl_model_path:str):
    """
    初始化SSL模型
    ssl_model_path: SSL模型路径
    is_half: 是否使用半精度
    device: 设备
    return:
    ssl_model: SSL模型
    """
    global ssl_model
    if ssl_model is None:
        cnhubert.cnhubert_base_path = ssl_model_path
        ssl_model = cnhubert.get_model()
        device = init_device()
        if is_half == True:
            ssl_model = ssl_model.half().to(device)
        else:
            ssl_model = ssl_model.to(device)
    return ssl_model


def get_bert_feature(text:str, word2ph:list, tokenizer, bert_model, device):
    """
    获取BERT特征
    text: 文本
    word2ph: 每个字转音素后，对应的个数，对于中文，就是声韵母，因此是全是 2 的 list
    tokenizer: 分词器
    bert_model: BERT模型
    device: 设备
    return:
    phone_level_feature: 音素级特征
    """
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T