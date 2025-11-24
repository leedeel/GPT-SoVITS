import re
from warnings import warn
import torch
import torchaudio
import torch.nn.functional as F
import librosa
import numpy as np
from feature_extractor import cnhubert
from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from api_interface.config import version, is_half, punctuation,cnhubert_path, bert_path
from GPT_SoVITS.common import (init_device,init_dict_language,init_bert_model,init_ssl_model,get_bert_feature)
from GPT_SoVITS.weights_manager import (change_sovits_weights)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
i18n=I18nAuto(language="zh_CN")
dtype=torch.float16 if is_half == True else torch.float32
dict_language=None
bert_model=None
ssl_model=None
tokenizer=None

def init():
    """
    初始化函数，用于加载模型等资源
    """
    print("init")
    global device,dict_language,tokenizer,bert_model,ssl_model
    # 初始化语言字典
    i18n,dict_language = init_dict_language(version=version)
    # 初始化BERT模型
    tokenizer,bert_model = init_bert_model(bert_model_path=bert_path)
    # 初始化ssl模型
    ssl_model = init_ssl_model(ssl_model_path=cnhubert_path)

# 初始化函数
init()


def clean_text_inf(text, language):
    """
    text: 字符串
    language: 所属语言
    
    return:
    phones: 音素 id 序列
    word2ph: 每个字转音素后，对应的个数，对于中文，就是声韵母，因此是全是 2 的 list
    norm_text: 归一化后文本
    """
    formattext = ""
    language = language.replace("all_","")
    for tmp in LangSegmenter.getTexts(text):
        if language == "ja":
            if tmp["lang"] == language or tmp["lang"] == "zh":
                formattext += tmp["text"] + " "
            continue
        if tmp["lang"] == language:
            formattext += tmp["text"] + " "
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    phones, word2ph, norm_text = clean_text(formattext, language)
    # print(f'音素: {phones}')
    phones = cleaned_text_to_sequence(phones)  # 统一了中、英、日等
    # print(f'音素 id: {phones}')
    return phones, word2ph, norm_text



def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(text=norm_text,
                                word2ph=word2ph,
                                tokenizer=tokenizer,
                                bert_model=bert_model,
                                device=device).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=dtype,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

@torch.no_grad()
def get_code_from_ssl(ssl,vq_model):
    ssl = vq_model.ssl_proj(ssl)
    quantized, codes, commit_loss, quantized_list = vq_model.quantizer(ssl)
    # print(codes.shape, codes.dtype)  # [n_q, B, T]
    return codes.transpose(0, 1)  # [B, n_q, T]


@torch.no_grad()
def get_code_from_wav(wav_path,vq_model):
    wav16k, sr = librosa.load(wav_path, sr=16000)
    if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
        # raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
        warn(i18n("参考音频在3~10秒范围外，请更换！"))
    wav16k = torch.from_numpy(wav16k)
    if is_half == True:
        wav16k = wav16k.half().to(device)
    else:
        wav16k = wav16k.to(device)
    ssl_content = ssl_model.model(wav16k.unsqueeze(0))[ 
        "last_hidden_state"
    ].transpose(
        1, 2
    )  # .float()
    codes = get_code_from_ssl(ssl_content,vq_model)  # [B, n_q, T]

    prompt_semantic = codes[0, 0] 
    return prompt_semantic


def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)
    # Merge punctuation into previous word
    for i in range(len(textlist)-1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i-1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i+1]:
            textlist[i] += textlist[i+1]
            del textlist[i+1]
            del langlist[i+1]
        else:
            i += 1

    return textlist, langlist


def nonen_clean_text_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text


def get_cleaned_text_final(text,language):
    if language in {"en","all_zh","all_ja"}:
        phones, word2ph, norm_text = clean_text_inf(text, language)
    elif language in {"zh", "ja","auto"}:
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    return phones, word2ph, norm_text


resample_transform_dict = {}
def _resample(audio_tensor, sr0, sr1, device):
    """
    重采样
    audio_tensor: 音频数据
    sr0: 原始采样率
    sr1: 目标采样率
    device: 设备
    return:
    audio_tensor: 重采样后的音频数据
    """
    print(f"重采样: {sr0} -> {sr1}")
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

def get_spepc(hps, filename, dtype, device, is_v2pro=False,target_dim=None):
    """
    获取音频特征
    hps: 超参数
    filename: 音频文件路径
    dtype: 数据类型
    device: 设备
    is_v2pro: 是否是 v2pro 版本
    target_dim: 目标维度
    return:
    spec: 音频特征
    audio: 音频数据
    """
    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = _resample(audio, sr0, sr1, device)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro == True:
        audio = _resample(audio, sr1, 16000, device).to(dtype)
     # 提取特征后检查维度
    if target_dim is not None and spec.shape[1] != target_dim:
        if spec.shape[1] > target_dim:
            spec = spec[:, :target_dim, :]
        else:
            pass
    return spec, audio



@torch.no_grad()
def get_vc_wav(
    sovits_path:str,
    source_wav_path:str,
    source_wav_text:str, 
    language:str, 
    ref_wav_path:str, 
    noise_scale=0.5):
    """ Voice Conversion
    sovits_path: sovits模型路径
    source_wav_path: 待变声的源音频
    source_wav_text: 对应文本
    language: 对应语言
    ref_wav_path: 目标人声
    noise_scale: 噪声比例
    """
    language = dict_language[language]

    phones, word2ph, norm_text = get_cleaned_text_final(source_wav_text, language)
    
    # 加载SoVITS模型权重
    ( version, model_version, if_lora_v3, vq_model, hps) = next(change_sovits_weights(sovits_path))
    print("使用SoVITS模型版本:", version, model_version, if_lora_v3)
    is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
    (spec, audio_len) = get_spepc(hps=hps, 
                                  filename=ref_wav_path,
                                  dtype=dtype,
                                  device=device,
                                  is_v2pro=is_v2pro)
    codes = get_code_from_wav(source_wav_path,vq_model)[None, None]  # 必须是 3D, [n_q, B, T]
    print(f"模型版本: {model_version}")
    print(f"频谱形状: {spec.shape}")
    print(f"vq_model是否有version属性: {hasattr(vq_model, 'version')}")
    if hasattr(vq_model, 'version'):
        print(f"vq_model.version: {vq_model.version}")
    
    # 根据模型版本调整输入维度
    if model_version != "v1":
        # 非v1版本需要704维输入
        if spec.dim() == 3:  # [B, D, T] 形状
            if spec.shape[1] > 704:  # 检查特征维度
                print(f"截断频谱维度: {spec.shape[1]} -> 704")
                spec = spec[:, :704, :]  # 取前704个特征维度
            elif spec.shape[1] < 704:
                print(f"警告: 频谱维度不足: {spec.shape[1]} < 704")
                # 可以考虑填充或使用其他处理
        else:
            print(f"警告: 意外的频谱形状: {spec.shape}")
    
    ge = vq_model.ref_enc(spec)  # [B, D, T/1] 
    quantized = vq_model.quantizer.decode(codes)  # [B, D, T]
    if hps.model.semantic_frame_rate == "25hz":
        quantized = F.interpolate(
            quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
        )
    _, m_p, logs_p, y_mask = vq_model.enc_p(
        quantized, torch.LongTensor([quantized.shape[-1]]), 
        torch.LongTensor(phones)[None], torch.LongTensor([len(phones)]), ge
    )
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = vq_model.flow(z_p, y_mask, g=ge, reverse=True)
    o = vq_model.dec((z * y_mask)[:, :, :], g=ge)  # [B, D=1, T], torch.float32 (-1, 1)
    audio = o.detach().cpu().numpy()[0, 0]    
    max_audio = np.abs(audio).max()  # 简单防止16bit爆音
    if max_audio > 1:
        audio /= max_audio
    yield hps.data.sampling_rate, (audio * 32768).astype(np.int16)
    
