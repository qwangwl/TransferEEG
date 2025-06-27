# -*- encoding: utf-8 -*-
'''
file       :trainer.py
Date       :2024/07/21 10:39:07
Author     :qwangwl
'''

import torch
import numpy as np
import copy
import utils

from trainers.BaseTrainer import BaseTrainer

# 定义trainer

class PRPLTrainer(BaseTrainer):
    def __init__(self,
                 model,
                 optimizer,
                 lr_scheduler=None,
                 n_epochs: int = 100,
                 batch_size: int = 96,
                 log_interval: int = 1,
                 early_stop: int = 0,
                 transfer_loss_weight: float = 1.0,
                 device: str = "cuda:0",
                 cluster_weight: int = 2,
                 **kwargs):
        super(PRPLTrainer, self).__init__(model=model,
                         optimizer=optimizer,
                         n_epochs=n_epochs,
                         batch_size=batch_size,
                         log_interval=log_interval,
                         early_stop=early_stop,
                         transfer_loss_weight=transfer_loss_weight,
                         lr_scheduler=lr_scheduler,
                         device=device,
                         **kwargs)
        
        self.cluster_weight = cluster_weight
        self.boost_factor = 0
        self.best_model_state = None
    
    def train_one_epoch(self, source_loader,  target_loader):
    
        self.model.train()
        
        # 定义每一个epoch跑多少个batch
        # 这主要是因为源域和目标域的数据量不一致
        len_source_loader = len(source_loader)
        len_target_loader = len(target_loader)
        n_batch = min(len_source_loader, len_target_loader) - 1 # 忽略掉最后一个batch，因为好像是因为最后一个batch太少了，对不上。

        # 定义一些损失的记录
        loss_clf = utils.AverageMeter()
        loss_cluster = utils.AverageMeter()
        loss_p = utils.AverageMeter()
        loss_transfer = utils.AverageMeter()
        
        # 定义迭代器
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)


        # self.boost_factor = self.cluster_weight * (epoch / self.n_epochs)
        # print(self.boost_factor)
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

            # 求解损失
            cls_loss, cluster_loss, P_loss, transfer_loss = self.model(src_data, tgt_data, src_label)

            loss = cls_loss + self.transfer_loss_weight * transfer_loss + 0.01 * P_loss + self.boost_factor * cluster_loss

            # 更新损失
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            # 更新各种记录
            loss_clf.update(cls_loss.detach().item())
            loss_cluster.update(cluster_loss.detach().item())
            loss_p.update(P_loss.detach().item())
            loss_transfer.update(transfer_loss.detach().item())

        return loss_clf.avg, loss_transfer.avg, loss_p.avg, loss_cluster.avg
    

    def train(self, source_loader, target_loader):

        stop = 0
        best_acc = 0.0
        log = []

        for epoch in range(self.n_epochs):
            self.model.train()

            # 每一轮epoch的训练
            loss_clf, loss_transfer, loss_p, loss_cluster \
                = self.train_one_epoch( source_loader, target_loader )
            
            # 每一个epoch后的操作
            self.epoch_based_processing(epoch=epoch, source_loader=source_loader)

            # 测试
            self.model.eval()
            with torch.no_grad():
                source_acc = self.test(source_loader)
                target_acc = self.test(target_loader)


            log.append([loss_clf, loss_transfer, loss_p, loss_cluster, source_acc, target_acc])

            info = (
                f'Epoch: [{epoch + 1:2d}/{self.n_epochs}], '
                f'loss_clf: {loss_clf:.4f}, '
                f'loss_transfer: {loss_transfer:.4f}, '
                f'loss_p: {loss_p:.4f}, '
                f'loss_cluster: {loss_cluster:.4f}, '
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

    def update_boost_factor(self, epoch):
        self.boost_factor = self.cluster_weight * ((epoch + 1) / self.n_epochs)

    def epoch_based_processing(self, epoch, source_loader):

        source_features = source_loader.dataset.data()
        source_labels = source_loader.dataset.label()

        # 首先更新boost_factor
        self.update_boost_factor(epoch)

        # 其次更新模型中的stroed_mat和cluster_label
        self.model.epoch_based_processing(epoch, source_features.to(self.device), source_labels.cuda().to(self.device))

    

