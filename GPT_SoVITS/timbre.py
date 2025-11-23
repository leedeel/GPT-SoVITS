import torch

def remove_timbre_features_advanced(refer, device, method="smoothing"):
    """
    增强版的音色特征去除函数
    
    参数:
        refer: 参考音频特征
        device: 计算设备
        method: 处理方法 ("normalization", "smoothing", "pca")
    
    返回:
        只包含节奏信息的特征
    """
    if not isinstance(refer, torch.Tensor):
        refer = torch.tensor(refer, device=device)
    else:
        refer = refer.to(device)
    
    if method == "normalization":
        # 方法1: 标准化 - 保留相对变化，去除绝对数值
        return normalize_features(refer)
    
    elif method == "smoothing":
        # 方法2: 时序平滑 - 保留整体趋势，去除细节（可能包含音色信息）
        return smooth_features(refer)
    
    # elif method == "pca":
    #     # 方法3: PCA降维重建 - 保留主要变化模式
    #     return pca_reconstruction(refer)
    
    else:
        return refer

def normalize_features(refer):
    """特征标准化"""
    if len(refer.shape) == 2:
        # 对每个特征维度独立标准化
        refer_norm = torch.zeros_like(refer)
        for i in range(refer.shape[1]):
            feature = refer[:, i]
            if torch.std(feature) > 1e-8:
                refer_norm[:, i] = (feature - torch.mean(feature)) / torch.std(feature)
            else:
                refer_norm[:, i] = feature
        return refer_norm
    else:
        # 对最后一个维度进行标准化
        mean = torch.mean(refer, dim=-1, keepdim=True)
        std = torch.std(refer, dim=-1, keepdim=True)
        return (refer - mean) / (std + 1e-8)

def smooth_features(refer, window_size=5):
    """时序平滑"""
    if len(refer.shape) == 2:
        # 一维平滑
        smoothed = torch.zeros_like(refer)
        for i in range(refer.shape[1]):
            feature = refer[:, i]
            # 使用简单的移动平均
            kernel = torch.ones(window_size, device=refer.device) / window_size
            smoothed_feature = torch.nn.functional.conv1d(
                feature.unsqueeze(0).unsqueeze(0), 
                kernel.unsqueeze(0).unsqueeze(0),
                padding=window_size//2
            ).squeeze()
            smoothed[:, i] = smoothed_feature
        return smoothed
    return refer