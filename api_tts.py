def validate_inputs(ref_wav_path, text):
    """
    验证输入参数是否有效
    
    Args:
        ref_wav_path: 参考音频路径
        text: 要合成的文本
        
    Raises:
        如果输入无效会显示警告
    """
    if not ref_wav_path:
        gr.Warning(i18n("请上传参考音频"))
    if not text:
        gr.Warning(i18n("请填入推理文本"))

def initialize_model_settings(model_version, prompt_text):
    """
    根据模型版本和提示文本来初始化设置
    
    Args:
        model_version: 模型版本
        prompt_text: 提示文本
        
    Returns:
        tuple: (ref_free, if_sr) 设置标志
    """
    ref_free = False
    if_sr = False
    
    if not prompt_text or len(prompt_text) == 0:
        ref_free = True
        
    if model_version in v3v4set:
        ref_free = False  # s2v3暂不支持ref_free
    else:
        if_sr = False
        
    # 清理不需要的模型以释放内存
    if model_version not in {"v3", "v4", "v2Pro", "v2ProPlus"}:
        clean_bigvgan_model()
        clean_hifigan_model()
        clean_sv_cn_model()
        
    return ref_free, if_sr

def process_reference_audio(ref_wav_path, pause_second, ref_free=False):
    """
    处理参考音频，提取特征
    
    Args:
        ref_wav_path: 参考音频路径
        pause_second: 停顿秒数
        ref_free: 是否不需要参考音频
        
    Returns:
        tuple: (prompt, zero_wav_torch, wav16k) 处理后的音频数据
    """
    # 创建静音片段
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half == True else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half == True:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)
    
    prompt = None
    wav16k = None
    
    if not ref_free:
        # 加载参考音频
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        
        # 验证音频长度
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            gr.Warning(i18n("参考音频在3~10秒范围外，请更换！"))
            raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            
        wav16k = torch.from_numpy(wav16k)
        if is_half == True:
            wav16k = wav16k.half().to(device)
        else:
            wav16k = wav16k.to(device)
            
        # 添加静音段
        wav16k = torch.cat([wav16k, zero_wav_torch])
        
        # 提取SSL特征
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)
    
    return prompt, zero_wav_torch, wav16k

def preprocess_text(text, text_language, how_to_cut):
    """
    预处理文本，包括切割和清理
    
    Args:
        text: 原始文本
        text_language: 文本语言
        how_to_cut: 切割方式
        
    Returns:
        list: 处理后的文本列表
    """
    # 根据指定方式切割文本
    cut_methods = {
        i18n("凑四句一切"): cut1,
        i18n("凑50字一切"): cut2,
        i18n("按中文句号。切"): cut3,
        i18n("按英文句号.切"): cut4,
        i18n("按标点符号切"): cut5
    }
    
    if how_to_cut in cut_methods:
        text = cut_methods[how_to_cut](text)
    
    # 清理多余的换行符
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
        
    print(i18n("实际输入的目标文本(切句后):"), text)
    
    # 分割文本并进一步处理
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    
    return texts

def prepare_text_features(prompt_text, prompt_language, text, text_language, ref_free, version):
    """
    准备文本的语音学和BERT特征
    
    Args:
        prompt_text: 提示文本
        prompt_language: 提示文本语言
        text: 目标文本
        text_language: 目标文本语言
        ref_free: 是否不需要参考
        version: 模型版本
        
    Returns:
        tuple: (bert, all_phoneme_ids, all_phoneme_len, norm_text2) 文本特征
    """
    # 获取提示文本的特征（如果不是ref_free模式）
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
    
    # 获取目标文本的特征
    phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
    print(i18n("前端处理后的文本(每句):"), norm_text2)
    
    # 合并特征
    if not ref_free:
        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
    else:
        bert = bert2
        all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)
    
    bert = bert.to(device).unsqueeze(0)
    all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
    
    return bert, all_phoneme_ids, all_phoneme_len, norm_text2

def generate_semantic_features(all_phoneme_ids, all_phoneme_len, prompt, bert, 
                              ref_free, top_k, top_p, temperature, 
                              max_sec, hz, cache_key=None, use_cache=False):
    """
    生成语义特征
    
    Args:
        all_phoneme_ids: 所有音素ID
        all_phoneme_len: 音素长度
        prompt: 提示特征
        bert: BERT特征
        ref_free: 是否不需要参考
        top_k: top_k采样参数
        top_p: top_p采样参数
        temperature: 温度参数
        max_sec: 最大秒数
        hz: 频率
        cache_key: 缓存键
        use_cache: 是否使用缓存
        
    Returns:
        tensor: 预测的语义特征
    """
    if use_cache and cache_key in cache:
        pred_semantic = cache[cache_key]
    else:
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            if use_cache:
                cache[cache_key] = pred_semantic
                
    return pred_semantic

def decode_audio_v2v3(pred_semantic, phones2, ref_wav_path, model_version, 
                     speed=1, inp_refs=None, sample_steps=8):
    """
    解码音频（针对v2/v3版本模型）
    
    Args:
        pred_semantic: 预测的语义特征
        phones2: 目标文本的音素
        ref_wav_path: 参考音频路径
        model_version: 模型版本
        speed: 语速
        inp_refs: 输入参考
        sample_steps: 采样步数
        
    Returns:
        tensor: 生成的音频
    """
    is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
    
    if model_version not in v3v4set:  # v1, v2系列
        refers = []
        sv_emb = []
        
        if is_v2pro:
            if sv_cn_model is None:
                init_sv_cn()
                
        # 处理输入参考
        if inp_refs:
            for path in inp_refs:
                try:
                    refer, audio_tensor = get_spepc(hps, path.name, dtype, device, is_v2pro)
                    refers.append(refer)
                    if is_v2pro:
                        sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor))
                except:
                    traceback.print_exc()
                    
        # 如果没有输入参考，使用主要参考音频
        if len(refers) == 0:
            refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device, is_v2pro)
            refers = [refers]
            if is_v2pro:
                sv_emb = [sv_cn_model.compute_embedding3(audio_tensor)]
                
        # 解码音频
        if is_v2pro:
            audio = vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), 
                refers, speed=speed, sv_emb=sv_emb
            )[0][0]
        else:
            audio = vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), 
                refers, speed=speed
            )[0][0]
    else:  # v3, v4系列
        refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
        phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
        
        # 使用CFM模型解码
        fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge=None, speed=speed)
        cfm_resss = []
        idx = 0
        
        # 分块处理
        Tref = 468 if model_version == "v3" else 500
        Tchunk = 934 if model_version == "v3" else 1000
        chunk_len = Tchunk - Tref
        
        while True:
            fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break
            idx += chunk_len
            
            # CFM推理
            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
            cfm_res = vq_model.cfm.inference(
                fea, torch.LongTensor([fea.size(1)]).to(fea.device), 
                mel2, sample_steps, inference_cfg_rate=0
            )
            cfm_res = cfm_res[:, :, mel2.shape[2]:]
            mel2 = cfm_res[:, :, -T_min:]
            fea_ref = fea_todo_chunk[:, :, -T_min:]
            cfm_resss.append(cfm_res)
            
        cfm_res = torch.cat(cfm_resss, 2)
        cfm_res = denorm_spec(cfm_res)
        
        # 使用声码器生成音频
        if model_version == "v3":
            if bigvgan_model is None:
                init_bigvgan()
            vocoder_model = bigvgan_model
        else:  # v4
            if hifigan_model is None:
                init_hifigan()
            vocoder_model = hifigan_model
            
        with torch.inference_mode():
            wav_gen = vocoder_model(cfm_res)
            audio = wav_gen[0][0]
            
    return audio

def post_process_audio(audio_opt, model_version, if_sr):
    """
    后处理音频，包括归一化和超分辨率
    
    Args:
        audio_opt: 原始音频
        model_version: 模型版本
        if_sr: 是否进行超分辨率
        
    Returns:
        tuple: (采样率, 处理后的音频数据)
    """
    # 确定输出采样率
    if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000  # v4
        
    # 应用超分辨率（如果需要）
    if if_sr and opt_sr == 24000:
        print(i18n("音频超分中"))
        audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
        max_audio = np.abs(audio_opt).max()
        if max_audio > 1:
            audio_opt /= max_audio
    else:
        audio_opt = audio_opt.cpu().detach().numpy()
        
    return opt_sr, (audio_opt * 32767).astype(np.int16)

# 重构后的主函数
def get_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
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
    文本到语音生成主函数
    
    Args:
        ref_wav_path: 参考音频路径
        prompt_text: 提示文本
        prompt_language: 提示文本语言
        text: 要合成的文本
        text_language: 文本语言
        how_to_cut: 文本切割方式
        top_k: top_k采样参数
        top_p: top_p采样参数
        temperature: 温度参数
        ref_free: 是否不需要参考音频
        speed: 语速
        if_freeze: 是否使用缓存
        inp_refs: 输入参考列表
        sample_steps: 采样步数
        if_sr: 是否进行超分辨率
        pause_second: 停顿秒数
        
    Yields:
        tuple: (采样率, 音频数据)
    """
    global cache
    
    # 1. 验证输入
    validate_inputs(ref_wav_path, text)
    
    # 2. 初始化模型设置
    ref_free, if_sr = initialize_model_settings(model_version, prompt_text)
    
    # 3. 处理参考音频
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
        
    text = text.strip("\n")
    print(i18n("实际输入的目标文本:"), text)
    
    prompt, zero_wav_torch, wav16k = process_reference_audio(
        ref_wav_path, pause_second, ref_free
    )
    t1 = ttime()
    
    # 4. 预处理文本
    texts = preprocess_text(text, text_language, how_to_cut)
    
    # 5. 逐句生成音频
    audio_opt = []
    timing_info = [t1 - t0]
    
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
        
    for i_text, current_text in enumerate(texts):
        if len(current_text.strip()) == 0:
            continue
            
        # 确保文本以合适的标点结尾
        if current_text[-1] not in splits:
            current_text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), current_text)
        
        # 准备文本特征
        bert, all_phoneme_ids, all_phoneme_len, norm_text2 = prepare_text_features(
            prompt_text, prompt_language, current_text, text_language, ref_free, version
        )
        
        t2 = ttime()
        
        # 生成语义特征
        pred_semantic = generate_semantic_features(
            all_phoneme_ids, all_phoneme_len, prompt, bert,
            ref_free, top_k, top_p, temperature, max_sec, hz,
            cache_key=i_text, use_cache=if_freeze
        )
        
        t3 = ttime()
        
        # 解码音频
        audio = decode_audio_v2v3(
            pred_semantic, phones2, ref_wav_path, model_version,
            speed, inp_refs, sample_steps
        )
        
        # 防止爆音
        max_audio = torch.abs(audio).max()
        if max_audio > 1:
            audio = audio / max_audio
            
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)  # 添加停顿
        
        t4 = ttime()
        timing_info.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()
        
    # 输出时间统计
    print("%.3f\t%.3f\t%.3f\t%.3f" % (
        timing_info[0], sum(timing_info[1::3]), 
        sum(timing_info[2::3]), sum(timing_info[3::3])
    ))
    
    # 6. 合并和后处理音频
    audio_opt = torch.cat(audio_opt, 0)
    opt_sr, final_audio = post_process_audio(audio_opt, model_version, if_sr)
    
    yield opt_sr, final_audio