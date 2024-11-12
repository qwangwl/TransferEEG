# -*- encoding: utf-8 -*-
'''
file       :Base.py
Date       :2024/07/22 11:57:17
Author     :qwangwl
'''

# 和Model_PR_PL相比，将分类器进行了修改。

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from loss_funcs import TransferLoss

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_1=64, hidden_2=64):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class LabelClassifier(nn.Module):
    def __init__(self, input_dim = 64, num_of_class=3):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_of_class)
    
    def forward(self, feature):
        y = self.fc1(feature)
        return self.fc2(y)
    
    def predict(self, feature):
        with torch.no_grad():
            logits = F.softmax(self.forward(feature), dim=1)
            y_preds = np.argmax(logits.cpu().numpy(), axis=1)
        return y_preds
    
    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1}
        ]
        return params


class Base(nn.Module):
    def __init__(self, 
                 input_dim: int= 310, 
                 num_of_class: int=3, 
                 max_iter: int=1000, 
                 transfer_loss_type: str="dann", 
                 **kwargs):
        super(Base, self).__init__()

        self.feature_extractor = FeatureExtractor(input_dim=input_dim)
        self.classifier = LabelClassifier(input_dim=64, num_of_class=num_of_class)

        self.max_iter = max_iter
        self.num_of_class = num_of_class
        self.transfer_loss_type = transfer_loss_type
        
        
        self.criterion = nn.CrossEntropyLoss()

        transfer_loss_args = {
            "loss_type" : self.transfer_loss_type,
            "max_iter" : self.max_iter,
            "num_class" : self.num_of_class,
            **kwargs
        }

        self.adv_criterion = TransferLoss(**transfer_loss_args)

    def forward(self, source, target, source_label):
        source_feature = self.feature_extractor(source)
        target_feature = self.feature_extractor(target)
        source_output = self.classifier(source_feature)

        cls_loss = self.criterion(source_output, source_label)

        kwargs = {}
        if self.transfer_loss_type == "lmmd":
            kwargs["source_label"] = source_label
            target_clf = self.classifier(target_feature)
            kwargs["target_logits"] = F.softmax(target_clf, dim=1)
        elif self.transfer_loss_type == "daan":
            source_clf = self.classifier(source_feature)
            kwargs['source_logits'] = F.softmax(source_clf, dim=1)
            target_clf = self.classifier(target_feature)
            kwargs['target_logits'] = F.softmax(target_clf, dim=1)

        trans_loss = self.adv_criterion(source_feature, target_feature, **kwargs)
        return cls_loss, trans_loss
    

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feature = self.feature_extractor(x)
            preds = self.classifier.predict(feature)
        return preds
    
    def predict_prob(self, x):
        self.eval()
        with torch.no_grad():
            feature = self.feature_extractor(x)
            output = self.classifier(feature)
            logits = F.softmax(output, dim=1)
        return logits

    def get_parameters(self):
        params = [
            *self.feature_extractor.get_parameters(),
            *self.classifier.get_parameters(),
        ]
        if self.transfer_loss_type == "dann":
            params.append(
                {"params": self.adv_criterion.loss_func.domain_classifier.parameters(), "lr_mult":1}
            )
        elif self.transfer_loss_type == "daan":
            params.append(
                {'params': self.adv_criterion.loss_func.domain_classifier.parameters(), "lr_mult":1}
            )
            params.append(
                {'params': self.adv_criterion.loss_func.local_classifiers.parameters(), "lr_mult":1}
            )
        # print(params)
        return params
    
    def epoch_based_processing(self, epoch_length):
        if self.transfer_loss_type == "daan":
            self.adv_criterion.loss_func.update_dynamic_factor(epoch_length)
    

    
