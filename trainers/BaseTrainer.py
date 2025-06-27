# -*- encoding: utf-8 -*-
'''
file       :BaseTrainer.py
Date       :2024/08/06 18:04:46
Author     :qwangwl
'''

# 定义基础的Trainer类

import torch
import numpy as np
import copy
import utils

# 定义trainer

class BaseTrainer(object):
    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler = None,
                 n_epochs: int = 100, 
                 log_interval: int = 1, 
                 early_stop: int = 0,
                 transfer_loss_weight: int = 1,
                 device: str = "cuda:0", 
                 **kwargs):
        """
        Initializes the BaseTrainer.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            n_epochs (int, optional): Number of epochs to train. Defaults to 100.
            log_interval (int, optional): Interval for logging training progress. Defaults to 1.
            early_stop (int, optional): Number of epochs without improvement before stopping. Defaults to 0.
            transfer_loss_weight (float, optional): Weight for the transfer loss. Defaults to 1.0.
            lr_scheduler (callable, optional): Learning rate scheduler. Defaults to None.
            device (str, optional): Device to use for training. Defaults to "cuda:0".
        """

        super(BaseTrainer, self).__init__()

        # 基础的模型
        self.model = model.to(device)
        self.optimizer = optimizer

        # 训练的参数
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.early_stop = early_stop

        # TODO 迁移的权重，后续也许要改，改成可变的，函数的
        self.transfer_loss_weight = transfer_loss_weight

        # 学习率调度器
        self.lr_scheduler = lr_scheduler

        # 其他的参数
        self.device = device
        self.best_model_state = None


    def get_model_state(self):
        """Get the current state of the model."""
        return self.model.state_dict()

    def get_best_model_state(self):
        """Get the best state of the model."""
        return self.best_model_state
    
    def train_one_epoch(self, source_loader,  target_loader):
    
        self.model.train()
        
        # 定义每一个epoch跑多少个batch
        # 这主要是因为源域和目标域的数据量不一致
        len_source_loader = len(source_loader)
        len_target_loader = len(target_loader)

        n_batch = min(len_source_loader, len_target_loader) - 1

        # 定义一些损失的记录
        loss_clf = utils.AverageMeter()
        loss_transfer = utils.AverageMeter()
        
        # 定义迭代器
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for _ in range(n_batch):

            # 获取数据
            try:
                src_data, src_label = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_data, src_label = next(source_iter)
            try:
                tgt_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_data, _ = next(target_iter)

            src_data, src_label = src_data.to(
                self.device), src_label.to(self.device)
            tgt_data = tgt_data.to(self.device)

            # 计算损失
            cls_loss, transfer_loss= self.model(src_data, tgt_data, src_label)
            loss = cls_loss + self.transfer_loss_weight * transfer_loss
            # print(cls_loss.size(), transfer_loss.size())
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            if self.lr_scheduler:
                self.lr_scheduler.step()

            # 更新各种记录
            loss_clf.update(cls_loss.detach().item())
            loss_transfer.update(transfer_loss.detach().item())
            
        return loss_clf.avg, loss_transfer.avg
    
    def train(self, source_loader, target_loader):

        stop = 0
        best_acc = 0.0
        log = []

        for epoch in range(self.n_epochs):
            self.model.train()

            # 每一轮epoch的训练
            loss_clf, loss_transfer \
                = self.train_one_epoch( source_loader, target_loader )
            
            # 做每一个epoch结束时的操作， 暂时只针对daan操作
            self.epoch_based_processing(epoch_length = min(len(source_loader), len(target_loader)) - 1)

            # 测试
            self.model.eval()
            with torch.no_grad():
                source_acc = self.test(source_loader)
                target_acc = self.test(target_loader)

            # 日志
            log.append([loss_clf, loss_transfer, source_acc, target_acc])

            # 保留信息，并在训练过程中输出
            info = (
                f'Epoch: [{epoch + 1:2d}/{self.n_epochs}], '
                f'loss_clf: {loss_clf:.4f}, '
                f'loss_transfer: {loss_transfer:.4f}, '
                f'source_acc: {source_acc:.4f}, '
                f'target_acc: {target_acc:.4f}'
            )

            # TODO 记录最佳结果
            stop += 1
            if target_acc > best_acc:
                best_acc = target_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                stop = 0

            # TODO 早停止
            if self.early_stop > 0 and stop >= self.early_stop:
                print(info)
                break

            # TODO 输出日志
            if (epoch+1) % self.log_interval == 0 or epoch == 0:
                print(info)
            

        np_log = np.array(log, dtype=float)
        return best_acc, np_log


    def test(self, dataloader):
        feature = dataloader.dataset.data()
        labels = dataloader.dataset.label()

        labels = np.argmax(labels.numpy(), axis=1)

        y_preds = self.model.predict(feature.to(self.device))
        acc = np.sum(y_preds == labels) / len(labels)
        return acc * 100.

    def epoch_based_processing(self, epoch_length):
        self.model.epoch_based_processing(epoch_length)

