import os
import os
import re
import traceback
import gradio as gr
import torch
import torchaudio
from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from api_interface.config import version, is_half, punctuation, cnhubert_path, bert_path
import librosa
import numpy as np
from time import time as ttime
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.common import (init_device,init_dict_language,init_bert_model,init_ssl_model)
from GPT_SoVITS.weights_manager import (change_gpt_weights,change_sovits_weights,DictToAttrRecursive)

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
    # 初始化设备参数
    device = init_device()
    # 初始化语言字典
    i18n,dict_language = init_dict_language(version=version)
    # 初始化BERT模型
    tokenizer,bert_model = init_bert_model(bert_model_path=bert_path)
    # 初始化ssl模型
    ssl_model = init_ssl_model(ssl_model_path=cnhubert_path)

# 初始化函数
init()


resample_transform_dict = {}


def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    # audio = load_audio(filename, int(hps.data.sampling_rate))

    # audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
    # audio = torch.FloatTensor(audio)

    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, device)
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
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio


def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


dtype = torch.float16 if is_half == True else torch.float32


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text



def get_bert_feature(text, word2ph):
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


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


def get_phones_and_bert(text, language, version, final=False):
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text,"ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text,"ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(dtype), norm_text


from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch

spec_min = -12
spec_max = 2


def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)
mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


sr_model = None


def audio_sr(audio, sr):
    global sr_model
    if sr_model == None:
        from tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(device, DictToAttrRecursive)
        except FileNotFoundError:
            gr.Warning(i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)


##ref_wav_path+prompt_text+prompt_language+text(单个)+text_language+top_k+top_p+temperature
# cache_tokens={}#暂未实现清理机制
from GPT_SoVITS.base_model_helper import (clear_all_models,init_model_by_version,get_all_models)
from GPT_SoVITS.timbre import (remove_timbre_features_advanced)


def get_tts_wav(
    sovits_path:str,
    gpt_path:str,
    ref_wav_path:str,
    prompt_text:str,
    prompt_language:str,
    text:str,
    text_language:str,
    how_to_cut=i18n("不切"),
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free=False,
    speed=1,
    if_freeze=False,
    inp_refs=None,
    sample_steps=8,
    if_sr=False,
    pause_second=0.3,
):
    """
    生成语音的主推理函数
    """
    # 1. 参数验证
    _validate_inputs(ref_wav_path, text)
    
    # 2. 初始化模型
    models = _initialize_models(sovits_path, gpt_path)
    version, model_version, if_lora_v3, vq_model, hps, hz, max_sec, t2s_model,hifigan_model, bigvgan_model, sv_cn_model = models
    
    # 3. 处理参考音频
    prompt_semantic, zero_wav_torch = _process_reference_audio(
        vq_model=vq_model,
        ref_wav_path=ref_wav_path,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        ref_free=ref_free,
        version=version,
        model_version=model_version,
        hps=hps,
        pause_second=pause_second,
    )
     
    # 4. 处理目标文本
    processed_texts = _process_target_text(
        text=text,
        how_to_cut=how_to_cut
    )
    
    # 5. 生成语音
    audio_opt = _generate_speech(
        ref_wav_path=ref_wav_path,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        processed_texts=processed_texts,
        prompt_semantic=prompt_semantic,
        vq_model=vq_model,
        hps=hps,
        t2s_model=t2s_model,
        sv_cn_model=sv_cn_model,
        version=version,
        model_version=model_version,
        hz=hz,
        max_sec=max_sec,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        if_freeze=if_freeze,
        inp_refs=inp_refs,
        speed=speed,
        zero_wav_torch=zero_wav_torch
    )
    
    # 6. 后处理
    final_audio, opt_sr = _post_process_audio(
        audio_opt, model_version, if_sr
    )
    
    yield opt_sr, final_audio


def _validate_inputs(ref_wav_path, text):
    """验证输入参数"""
    if not ref_wav_path:
        raise Exception(i18n("请上传参考音频"))
    if not text:
        raise Exception(i18n("请填入推理文本"))


def _initialize_models(sovits_path, gpt_path):
    """初始化所有需要的模型"""
    # 加载SoVITS模型权重
    version, model_version, if_lora_v3, vq_model, hps = next(change_sovits_weights(sovits_path))
    print("使用SoVITS模型版本:", version, model_version, if_lora_v3)
    
    # 加载GPT模型权重
    hz, max_sec, t2s_model = next(change_gpt_weights(gpt_path))
    
    # 加载基础模型
    init_model_by_version(version=model_version)
    
    # 获取基础模型实例
    hifigan_model, bigvgan_model, sv_cn_model = get_all_models()
    print(f"使用基础模型实例:hifigan_model:{hifigan_model}, bigvgan_model:{bigvgan_model}, sv_cn_model:{sv_cn_model}")
    
    return version, model_version, if_lora_v3, vq_model, hps, hz, max_sec, t2s_model, hifigan_model, bigvgan_model, sv_cn_model


def _process_reference_audio(vq_model,ref_wav_path, prompt_text, prompt_language, ref_free, 
                           version, model_version, hps, pause_second):
    """处理参考音频，提取语义特征"""
    # 初始化零音频
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half == True else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half == True:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)
        
    prompt_semantic = None
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
        
        # 提取参考音频的语义特征
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            if is_half == True:
                wav16k = wav16k.half().to(device)
            else:
                wav16k = wav16k.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)
    
    return prompt_semantic, zero_wav_torch


def _process_target_text(text, how_to_cut):
    """处理目标文本，包括切分和格式化"""
    text = text.strip("\n")
    print("实际输入的目标文本:", text)
    
    # 文本切分
    cut_functions = {
        i18n("凑四句一切"): cut1,
        i18n("凑50字一切"): cut2,
        i18n("按中文句号。切"): cut3,
        i18n("按英文句号.切"): cut4,
        i18n("按标点符号切"): cut5,
    }
    
    if how_to_cut in cut_functions:
        text = cut_functions[how_to_cut](text)
    
    # 处理连续换行
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    
    return texts


def _generate_speech(ref_wav_path,
                     prompt_text,
                     prompt_language,
                     processed_texts, 
                     prompt_semantic, 
                     vq_model, hps, 
                     t2s_model,sv_cn_model, version, model_version, hz, max_sec, top_k, top_p, 
                    temperature, if_freeze, inp_refs, speed, zero_wav_torch):
    """生成语音波形"""
    cache = {}
    audio_opt = []
    
    phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
    
    for i_text, text in enumerate(processed_texts):
        # 跳过空行
        if len(text.strip()) == 0:
            continue
            
        # 处理单句文本
        audio_segment = _process_single_sentence(
            ref_wav_path=ref_wav_path,
            text=text,
            i_text=i_text,
            prompt_semantic=prompt_semantic,
            vq_model=vq_model,
            hps=hps,
            t2s_model=t2s_model,
            sv_cn_model=sv_cn_model,
            version=version,
            model_version=model_version,
            hz=hz,
            max_sec=max_sec,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            if_freeze=if_freeze,
            inp_refs=inp_refs,
            speed=speed,
            zero_wav_torch=zero_wav_torch,
            phones1=phones1,
            bert1=bert1,
            cache=cache
        )
        
        if audio_segment is not None:
            audio_opt.append(audio_segment)
            audio_opt.append(zero_wav_torch)  # 添加停顿
    
    # 合并所有音频片段
    if audio_opt:
        return torch.cat(audio_opt, 0)
    else:
        return torch.tensor([])


def _process_single_sentence(ref_wav_path, text, i_text, prompt_semantic, vq_model, hps, 
                           t2s_model, sv_cn_model,version, model_version, hz, max_sec, top_k, top_p, 
                           temperature, if_freeze, inp_refs, speed, zero_wav_torch, 
                           phones1, bert1, cache):
    """处理单个句子并生成对应的音频"""
    # 确保句子以标点结尾
    if text[-1] not in splits:
        text += "。"
    
    print("实际输入的目标文本(每句):", text)
    
    # 获取文本的音素和BERT特征
    phones2, bert2, norm_text2 = get_phones_and_bert(text, "all_zh", version)
    print(i18n("前端处理后的文本(每句):"), norm_text2)
    
    # 准备BERT输入和音素ID
    bert = torch.cat([bert1, bert2], 1)
    all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
    bert = bert.to(device).unsqueeze(0)
    all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
    
    # 生成语义特征
    pred_semantic = _generate_semantic_features(
        all_phoneme_ids=all_phoneme_ids,
        all_phoneme_len=all_phoneme_len,
        prompt_semantic=prompt_semantic,
        bert=bert,
        t2s_model=t2s_model,
        hz=hz,
        max_sec=max_sec,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        if_freeze=if_freeze,
        i_text=i_text,
        cache=cache
    )
    
    if pred_semantic is None:
        return None
    
    # 获取参考音频特征
    refers, sv_emb = _get_reference_features(
        sv_cn_model=sv_cn_model,
        hps=hps,
        inp_refs=inp_refs,
        ref_wav_path=ref_wav_path,
        device=device)
    
    # 解码生成音频
    audio = vq_model.decode(
        pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), 
        refers, speed=speed, sv_emb=sv_emb
    )[0][0]
    
    # 防止爆音
    max_audio = torch.abs(audio).max()
    if max_audio > 1:
        audio = audio / max_audio
    
    return audio


def _generate_semantic_features(all_phoneme_ids, all_phoneme_len, prompt_semantic, 
                              bert, t2s_model, hz, max_sec, top_k, top_p, 
                              temperature, if_freeze, i_text, cache):
    """生成语义特征"""
    # 检查缓存
    if i_text in cache and if_freeze == True:
        return cache[i_text]
    
    # 生成新的语义特征
    with torch.no_grad():
        pred_semantic, idx = t2s_model.model.infer_panel(
            all_phoneme_ids,
            all_phoneme_len,
            prompt_semantic,
            bert,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=hz * max_sec,
        )
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        
        # 更新缓存
        if if_freeze:
            cache[i_text] = pred_semantic
    
    return pred_semantic


def _get_reference_features(sv_cn_model,hps, inp_refs, ref_wav_path, device):
    """获取参考音频的声学特征和说话人嵌入"""
    refers = []
    sv_emb = []
    
    # 处理额外参考音频
    if inp_refs:
        for path in inp_refs:
            try:
                refer, audio_tensor = get_spepc(hps, path.name, dtype, device, True)
                refers.append(refer)
                sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor))
            except:
                traceback.print_exc()
    
    # 如果没有额外参考音频，使用主参考音频
    if len(refers) == 0:
        refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device, True)
        refers.append(refer)
        sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor))
    
    return refers, sv_emb


def _post_process_audio(audio_opt, model_version, if_sr):
    """对生成的音频进行后处理"""
    # 确定输出采样率
    sr_mapping = {
        "v1": 32000,
        "v2": 32000,
        "v2Pro": 32000,
        "v2ProPlus": 32000,
        "v3": 24000,
        "v4": 48000
    }
    opt_sr = sr_mapping.get(model_version, 48000)
    
    # 音频超分处理
    if if_sr == True and opt_sr == 24000:
        print(i18n("音频超分中"))
        audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
        max_audio = np.abs(audio_opt).max()
        if max_audio > 1:
            audio_opt /= max_audio
        final_audio = (audio_opt * 32767).astype(np.int16)
    else:
        # 转换为numpy数组并量化
        audio_np = audio_opt.cpu().detach().numpy()
        final_audio = (audio_np * 32767).astype(np.int16)
    
    return final_audio, opt_sr

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


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split("(\d+)", s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text
