# -*- encoding: utf-8 -*-
'''
file       :base.py
Date       :2024/10/20 14:30:28
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

# 跑一下基本的模型，做一个基线

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from torch import nn
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing

from config import get_parser
from datasets import SEEDFeatureDataset
from models import Base, PRPL
from trainers import BaseTrainer, PRPLTrainer

def setup_seed(seed):  ## setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weight_init(m):  ## model parameter intialization
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        m.bias.data.zero_()


class CustomDataset(TensorDataset):
    def __init__(self, d1, d2):
        super(CustomDataset, self).__init__()
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        return len(self.d1)
    
    def __getitem__(self, idx):
        return self.d1[idx], self.d2[idx]
    
    def data(self):
        return self.d1
    def label(self):
        return self.d2

def load_seed(args, target, source_lists = None):
    if args.dataset_name == "seed3":
        setattr(args, "num_of_class", 3)
        setattr(args, "path", args.seed3_path)
    elif args.dataset_name == "seed4":
        setattr(args, "num_of_class", 4)
        setattr(args, "path", args.seed4_path)

    EEG, Label, Group = SEEDFeatureDataset(args.path, session=args.session).data()
    Label += 1
    EEG = EEG.reshape(-1, 310)
    tGroup = Group[:, 2] - 1 # 影片的group
    sGroup = Group[:, 1]
    

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(sGroup):
        EEG[sGroup==i] = min_max_scaler.fit_transform(EEG[sGroup == i])

    one_hot_mat = np.eye(len(Label), 3)[Label].astype("float32")

    # 源域数据，如果只有几个受试者
    if source_lists is None:
        source_lists = list(range(1, 16))
        source_lists.remove(target)
    source_lists = np.array(source_lists)
    
    # 获得目标域数据
    target_features = torch.from_numpy(EEG[sGroup==target]).type(torch.Tensor)
    target_labels = torch.from_numpy(one_hot_mat[sGroup==target])
    torch_dataset_target = CustomDataset(target_features, target_labels)
    
    # 获得源域数据
    source_features = torch.from_numpy(EEG[np.isin(sGroup, source_lists)]).type(torch.Tensor)
    source_labels = torch.from_numpy(one_hot_mat[np.isin(sGroup, source_lists)])
    torch_dataset_source = CustomDataset(source_features, source_labels)

    return torch_dataset_source, torch_dataset_target

def get_model_utils(args):
    # 模型
    base_params = {
        "num_of_class" : args.num_of_class,
    }
    params = {
        "transfer_loss_type" : args.transfer_loss_type,
        "max_iter" : args.max_iter
    }

    combined_params = {**base_params, **params}
    model = Base(**combined_params).cuda()

    # 优化器
    params = model.get_parameters()
    optimizer = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)

    # 学习率scheduler
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        scheduler = None
    # 训练器
    trainer_params = {
        "lr_scheduler" : scheduler,
        "batch_size" : args.batch_size,
        "n_epochs" : args.n_epochs,
        "transfer_loss_weight" : args.transfer_loss_weight,
        "early_stop" : args.early_stop,
        "tmp_saved_path" : args.tmp_saved_path,
        "log_interval" : args.log_interval,
    }
    trainer = None
    trainer = BaseTrainer(
        model, 
        optimizer, 
        **trainer_params
    )
    return trainer


def train(target, args):

    # 每一个受试者都重新定义seed
    setup_seed(args.seed)
    
    cur_target_saved_path = os.path.join(args.tmp_saved_path, str(target))
    create_dir_if_not_exists(cur_target_saved_path)

    # 获取数据
    source_lists = list(range(1, 16))
    source_lists.remove(target)

    dataset_source, dataset_target = load_seed(args, target=target, \
                                               source_lists=source_lists )
    loader_source = DataLoader(
            dataset=dataset_source,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
            )
    loader_target = DataLoader(
            dataset=dataset_target,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
            )

    # 定义max_iter
    setattr(args, "max_iter", 1000) # 按理说应该等于上式，但是原始代码定义为了n_epochs

    # 获得训练器
    trainer = get_model_utils(args)
    
    # 训练
    best_acc, np_log = trainer.train(loader_source, loader_target)    
    
    # 保存模型
    if args.saved_model:
        torch.save(trainer.get_model_state(), os.path.join(cur_target_saved_path, f"many_last.pth"))
        torch.save(trainer.get_best_model_state(), os.path.join(cur_target_saved_path, f"many_best.pth"))
    np.savetxt(os.path.join(args.tmp_saved_path, f"t{target}.csv"), np_log, delimiter=",",  fmt='%.4f')
    return best_acc



def main(args):
    setup_seed(args.seed)
    # 用来测试不同的迁移损失函数

    best_acc_mat = []
    for target in range(1, 16):
        best_acc = train(target, args)
        best_acc_mat.append(best_acc)
        print(f"target: {target}, best_acc: {best_acc}")
    
    mean = np.mean(best_acc_mat)
    std = np.std(best_acc_mat)

    for target, best_acc in enumerate(best_acc_mat):
        print(f"target: {target+1}, best_acc: {best_acc:.6f}")
    print(f"all_best_acc: {mean:.4f} ± {std:.4f}")


def create_dir_if_not_exists(path):
    # 构建完整的路径
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    main(args)

