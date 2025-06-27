# -*- encoding: utf-8 -*-
'''
file       :PRPL.py
Date       :2024/08/07 11:04:39
Author     :qwangwl
'''

# PRPL的模型实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from loss_funcs import PairLoss, TransferLoss

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
    def __init__(self, 
                 num_of_class: int = 3, 
                 low_rank: int = 32, 
                 max_iter: int = 1000, 
                 upper_threshold: float = 0.9, 
                 lower_threshold: float = 0.5):
        
        super(LabelClassifier, self).__init__()
        
        # 定义可训练参数
        self.U = nn.Parameter(torch.randn(low_rank, 64), requires_grad=True)
        self.V = nn.Parameter(torch.randn(low_rank, 64), requires_grad=True)
        self.P = torch.randn(num_of_class, 64)
        self.stored_mat = torch.matmul(self.V, self.P.T)

        # 定义参数
        self.max_iter = max_iter
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.threshold = upper_threshold
        self.num_of_class = num_of_class

        self.cluster_label = np.zeros(num_of_class)
        

    def forward(self, feature):
        preds = torch.matmul(torch.matmul(self.U, feature.T).T, self.stored_mat)   
        logits = F.softmax(preds, dim=1)
        return logits 

    def update_P(self, source_feature, source_label):
        self.P = torch.matmul(torch.inverse(torch.diag(source_label.sum(axis=0)) + torch.eye(self.num_of_class).cuda()),
                              torch.matmul(source_label.T, source_feature))
        self.stored_mat = torch.matmul(self.V, self.P.T)

    
    def update_cluster_label(self, source_feature, source_label):
        self.eval()
        with torch.no_grad():
            logits = self.forward(source_feature)
            source_cluster = np.argmax(logits.cpu().detach().numpy(), axis=1)
            source_label = np.argmax(source_label.cpu().numpy(), axis=1)
            for i in range(self.num_of_class):
                # 获取当前聚簇的样本
                samples_in_cluster_index = np.where(source_cluster == i)[0]
                label_for_samples = source_label[samples_in_cluster_index]

                # 对当前聚簇的样本进行分析，获得该聚簇类别数最多的类别，并定义
                if len(label_for_samples) == 0:
                    self.cluster_label[i] = 0
                else:
                    label_for_current_cluster = np.argmax(np.bincount(label_for_samples))
                    self.cluster_label[i] = label_for_current_cluster

    def predict(self, feature):
        self.eval()
        with torch.no_grad():
        
            logits = self.forward(feature)
            cluster = np.argmax(logits.cpu().numpy(), axis=1)

            preds = np.array([self.cluster_label[i] for i in cluster])
        return preds
    
    def get_parameters(self):
        params = [
            {"params": self.U, "lr_mult": 1},
            {"params": self.V, "lr_mult": 1},
        ]
        return params



class PRPL(nn.Module):
    def __init__(self, 
                 num_of_class: int = 3, 
                 max_iter: int = 1000,
                 low_rank: int = 32, 
                 upper_threshold: float = 0.9, 
                 lower_threshold: float = 0.5,
                 **kwargs):
        super(PRPL, self).__init__()
        self.max_iter = max_iter
        self.feature_extractor = FeatureExtractor(310, 64, 64)

        # 去掉下面那个代码会导致降低1.2%的准确率，属于噪声叭
        # self.fea_extrator_g = FeatureExtractor(310, 64, 64) #这个会有影响
        self.classifier = LabelClassifier(num_of_class = num_of_class, 
                                               low_rank = low_rank, 
                                               max_iter = max_iter, 
                                               upper_threshold = upper_threshold, 
                                               lower_threshold = lower_threshold)
        
        self.pair_loss = PairLoss(max_iter=max_iter)
        self.transfer_loss = TransferLoss(loss_type="dann", max_iter=max_iter)

    def forward(self, source, target, source_label):

        batch_size = source.size(0)
        source_feature = self.feature_extractor(source)
        target_feature = self.feature_extractor(target)

        self.classifier.update_P(self.feature_extractor(source), source_label)
        # 直接使用source_feature和额外的进行一次特征的提取准确率会不一致，且准确率会下降
        # self.classifer.update_P(source_feature, source_label)
        
        source_logits = self.classifier(source_feature)
        target_logits = self.classifier(target_feature)

        clf_loss, cluster_loss = self.pair_loss(source_label, source_logits, target_logits)
        
        P_loss=torch.norm(torch.matmul(self.classifier.P.T,self.classifier.P)-torch.eye(64).to(source.device),'fro')

        trans_loss = self.transfer_loss(source_feature+0.005*torch.randn((batch_size,64)).to(source_feature.device), \
                                         target_feature+0.005*torch.randn((batch_size,64)).to(target_feature.device))

        return clf_loss, cluster_loss,  P_loss, trans_loss
    

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feature = self.feature_extractor(x)
            preds = self.classifier.predict(feature)
        return preds
    
    def predict_prob(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.classifier(self.feature_extractor(x)).cpu().numpy()
            cluster_label = self.classifier.cluster_label.astype(np.int8)
            logits[:, cluster_label] = logits[:, [0, 1, 2]]
        return logits

    def get_parameters(self) :
        params = [
            *self.feature_extractor.get_parameters(),
            *self.classifier.get_parameters(),
            {"params": self.transfer_loss.loss_func.domain_classifier.parameters(), "lr_mult":1}
        ]
        return params
    
    def epoch_based_processing(self, epoch, source_features, source_labels):
        
        self.pair_loss.update_threshold(epoch)
        self.classifier.update_cluster_label(self.feature_extractor(source_features), source_labels)
