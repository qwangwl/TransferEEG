# -*- encoding: utf-8 -*-
'''
file       :transfer_losses.py
Date       :2024/08/04 11:29:28
Author     :qwangwl
'''

# 对所有关于迁移学习损失的封装

import torch
from torch import nn
from loss_funcs.transfer_loss_funcs import *

class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super(TransferLoss, self).__init__()

        if loss_type == "dann":
            self.loss_func = AdversarialLoss(**kwargs)
        elif loss_type == "mmd":
            self.loss_func = MMDLoss(**kwargs)
        elif loss_type == "coral":
            self.loss_func = CORAL
        elif loss_type == "daan":
            self.loss_func = DAANLoss(**kwargs)
        # elif loss_type == "lmmd":
        #     self.loss_func = LMMDLoss(**kwargs)
        # lmmd暂时有点问题，先不用了
        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda x, y : torch.tensor(0.0, device=x.device)

    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)
    
    def get_parameters(self):
        params = []
        if self.loss_type == "dann":
            params.append(
                {"params": self.loss_func.domain_classifier.parameters(), "lr_mult":1}
            )
        elif self.loss_type == "daan":
            params.append(
                {'params': self.loss_func.domain_classifier.parameters(), "lr_mult":1}
            )
            params.append(
                {'params': self.loss_func.local_classifiers.parameters(), "lr_mult":1}
            )
        return params