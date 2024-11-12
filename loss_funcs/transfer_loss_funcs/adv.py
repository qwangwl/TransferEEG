import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb
    
    def step(self):
        self.curr_iter = self.curr_iter + 1

class AdversarialLoss(nn.Module):
    '''
    Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
    '''
    def __init__(self, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, hidden_1=64, **kwargs):
        super(AdversarialLoss, self).__init__()
        self.domain_classifier = discriminator(hidden_1=hidden_1)
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)
        
    def forward(self, source, target):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()

        #! 有两种实现方法，这种方法是由wangjindong github中实现给出的
        # source_loss = self.get_adversarial_result(source, True, lamb)
        # target_loss = self.get_adversarial_result(target, False, lamb)
        # adv_loss = 0.5 * (source_loss + target_loss)
        # return adv_loss
        
        #! 有两种实现方法 下面这种是参考PRPL代码给出的
        # ? 这种方法要比上面那个方法的准确率提高了2%。我无法理解这一现象。
        return self.get_adversarial_result_by_prpl(source, target, lamb)

    def get_adversarial_result_by_prpl(self, source, target, lamb):
        f = ReverseLayerF.apply(torch.cat((source, target), dim=0), lamb)
        d = self.domain_classifier(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((source.size(0), 1)).to(source.device)
        d_label_t = torch.zeros((target.size(0), 1)).to(target.device)
        loss_fn = nn.BCELoss(reduction="mean")
        return 0.5 * (loss_fn(d_s, d_label_s) + loss_fn(d_t, d_label_t))
    
    
    def get_adversarial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        if source:
            domain_label = torch.ones(len(x), 1).long()
        else:
            domain_label = torch.zeros(len(x), 1).long()
        loss_fn = nn.BCELoss(reduction="mean")
        loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv
    
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class discriminator(nn.Module):
    def __init__(self, hidden_1=64):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params