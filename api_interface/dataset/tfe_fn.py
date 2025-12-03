import os
from transformers import AutoModelForMaskedLM, AutoTokenizer
from api_interface.config import (is_half,bert_path)
from GPT_SoVITS.text.cleaner import clean_text
import torch
import json
from api_interface.dataset.common import get_device,get_bert_dir,save_pth,get_tfe_file_path

language_v1_to_language_v2 = {
    "ZH": "zh",
    "zh": "zh",
    "JP": "ja",
    "jp": "ja",
    "JA": "ja",
    "ja": "ja",
    "EN": "en",
    "en": "en",
    "En": "en",
    "KO": "ko",
    "Ko": "ko",
    "ko": "ko",
    "yue": "yue",
    "YUE": "yue",
    "Yue": "yue",
}


def process_tfe(wav_path:str, 
                text:str, 
                language:str,
                version:str,
                device:str,
                bert_dir:str,
                bert_model:any,
                tokenizer:any,
                )->list:
    """
    处理单个音频,收集特征
    参数:
    wav_path: 音频路径
    text: 音频文本
    language: 音频语言
    version: 版本号
    tfe_dir: 文本特征存储路径
    bert_model: bert模型
    tokenizer: bert分词器
    返回:
    特征列表
    """
    try:
        wav_name = os.path.basename(wav_path)
        print(f"正在提取{wav_name}的文本特征")
        corrected_text = text.replace("%", "-").replace("￥", ",")
        phones, word2ph, norm_text = clean_text(text=corrected_text, 
                                                language=language, 
                                                version=version)
        path_bert = "%s/%s.pt" % (bert_dir, wav_name)
        if os.path.exists(path_bert) == False and language == "zh":
            bert_feature = get_bert_feature(bert_model=bert_model,
                                            tokenizer=tokenizer,
                                            text=norm_text,
                                            device=device,
                                            word2ph=word2ph)
            assert bert_feature.shape[-1] == len(phones)
            save_pth(bert_feature, path_bert)
            phones = " ".join(phones)
        return [wav_name, phones, word2ph, norm_text]
    except Exception as e:
        print(f"{wav_name}提取文本特征失败: {e}")
        return None



def get_bert_feature(bert_model, tokenizer, text, word2ph,device):
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



def train_tfe(version:str,
              opt_dir:str,
              train_dataset_list:list[dict[str, str]],
              language:str):
    """
    文本分词与特征提取
    参数:
    version: 版本号,可选值:["v1", "v2", "v4", "v2Pro", "v2ProPlus"]
    train_dataset_list: 训练数据集列表,每个元素为一个字典,包含以下字段:
    - text: 音频文本
    - wav_path: 音频路径
    
    """
    if not os.path.exists(bert_path):
        raise FileNotFoundError(bert_path)
    # 1.文本分词特征提取
    tfe_process_file_path = get_tfe_file_path(opt_dir=opt_dir)
    if os.path.exists(tfe_process_file_path):
        return tfe_process_file_path
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    device = get_device()
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)
    todo_list = []
    # 向每个gpu分配若干任务
    for dataset in train_dataset_list:
        try:
            text = dataset.get("text")
            wav_path = dataset.get("wav_path")
            if language in language_v1_to_language_v2.keys():
                todo.append([wav_path, text, language_v1_to_language_v2.get(language, language)])
            else:
                print(f"[Waring] The {language = } of {wav_path} is not supported for training.")
        except Exception as e:
            print(f"{dataset}加入训练任务失败: {e}")
            raise e
    
    bert_dir = get_bert_dir(opt_dir=opt_dir)
    result_list = []
    for todo in todo_list:
        wav_path, text, lan = todo
        process_result = process_tfe(wav_path=wav_path, 
                                     text=text, 
                                     language=lan,
                                     version=version,
                                     device=device,
                                     bert_dir=bert_dir,
                                     bert_model=bert_model,
                                     tokenizer=tokenizer)
        if process_result is not None:
            result_list.append(process_result)
    with open(tfe_process_file_path, "w", encoding="utf8") as f:
        f.write(json.dumps(result_list, ensure_ascii=False, indent=4))
    return tfe_process_file_path
    