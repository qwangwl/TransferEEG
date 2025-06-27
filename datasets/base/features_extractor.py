# -*- encoding: utf-8 -*-
'''
file       :feature_extractor.py
Date       :2025/02/12 17:21:44
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''
from datasets.features import PSD, DE

class FeatureExtractor(object):
    
    def __init__(self, feature, **kwargs):
        super(FeatureExtractor, self).__init__()
        if feature == "psd":
            self.feature_func = PSD(**kwargs)
        elif feature == "de":
            self.feature_func = DE(**kwargs)
        else:
            self.feature_func = lambda x: x

    def __call__(self, data, **kwargs):
        return self.feature_func(data, **kwargs)
    
    def feature_dim(self):
        if hasattr(self.feature_func, "feature_dim"):
            return self.feature_func.feature_dim()
        else:
            # Return None to let the caller handle feature computation
            return None

