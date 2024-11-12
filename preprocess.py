# -*- encoding: utf-8 -*-
'''
file       :preprocess.py
Date       :2024/10/31 20:28:27
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''
import numpy as np


import numpy as np

def augmentation(feature, label, group, alpha=0.5):
    augment_data = []
    augment_label = []

    for g in np.unique(group):
        video_feature = feature[group.flatten() == g]
        video_label = label[group.flatten() == g]
        
        # 随机选择样本索引
        indices = np.random.randint(0, len(video_feature), (len(video_feature), 2))
        # 生成 lam 值
        lam = np.random.beta(alpha, alpha, size=len(video_feature)).reshape(-1, 1)
        # 计算加权特征
        weighted_features = lam * video_feature[indices[:, 0]] + (1 - lam) * video_feature[indices[:, 1]]
        
        augment_data.append(weighted_features)
        augment_label.extend(video_label)

    return np.vstack(augment_data), np.array(augment_label)

# 示例用法
# augment_data, augment_label = augmentation(feature, label, group)
