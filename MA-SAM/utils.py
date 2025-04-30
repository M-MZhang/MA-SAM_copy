import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
import pickle
import math
from torch.optim.lr_scheduler import LambdaLR
import nibabel as nib
import json 
import errno
from monai.metrics import compute_hausdorff_distance

def hd_score(p, y):

    tmp_hd = compute_hausdorff_distance(p, y) # HD
    tmp_hd = torch.mean(tmp_hd)

    return tmp_hd

class HD_Score(nn.Module):
    def __init__(self, n_classes):
        super(HD_Score, self).__init__()
        self.smooth = 1e-5
        self.hd = compute_hausdorff_distance
        self.n_classes = n_classes
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = 1.0*(input_tensor == i)  # * torch.ones_like(input_tensor)
            temp_prob[input_tensor == -100] = -100
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def forward(self, inputs, target, softmax=False):
      
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        hd = self.hd(inputs.unsqueeze(1), target.unsqueeze(1)).mean() 
        hd = torch.mean(hd)
     
        return hd.item()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = 1.0*(input_tensor == i)  # * torch.ones_like(input_tensor)
            temp_prob[input_tensor == -100] = -100
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        mask = (target != -100)
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if self.n_classes > 1:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
            # weight = [0.5,1.5]
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth  # 平滑项，防止除零

    def forward(self, y_pred, y_true):
        # y_pred: 模型输出的概率 [N, H, W]（未经过sigmoid）
        # y_true: 真实标签 [N, H, W]，值为0或1
        y_pred = torch.sigmoid(y_pred)  # 转换为概率 [0,1]
        if (math.nan in y_pred) or (math.inf in y_pred):
            print("Erro!")
        
        # 展平张量
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        # 计算交集和并集
        intersection = (y_pred_flat * y_true_flat).sum()
        union = y_pred_flat.sum() + y_true_flat.sum()
        
        # Dice Loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
#         # 计算概率 p_t
#         pt = torch.exp(-bce_loss)  # p_t = sigmoid(logit) for y=1, 1-sigmoid(logit) for y=0
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

class MultiClassFocalLoss(nn.Module):   
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha  # 可传入类别权重列表（如 [0.1, 0.9]）
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 计算 p_t = softmax(output)[target_class]
        
        if self.alpha is not None:
            alpha = self.alpha[targets]  # 按 target 选择 alpha
            fl_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            fl_loss = (1 - pt) ** self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return fl_loss.mean()
        elif self.reduction == 'sum':
            return fl_loss.sum()
        else:
            return fl_loss

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1

class IoU(nn.Module):
    
    def __init__(self, reduction='mean'):
        super(IoU, self).__init__()
        self.reduction = reduction
 
    def leave_only_batch_and_flatten(self, inputs, targets):
        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        return inputs, targets
 
    def forward(self, inputs, targets, smooth=1):
 
        inputs, targets = self.leave_only_batch_and_flatten(inputs, targets)
        # inputs_after_sigmoid = torch.sigmoid(inputs)
 
        intersection = (inputs * targets).sum(1)
        total = (inputs + targets).sum(1)
        union = total - intersection
 
        IoU = (intersection + smooth)/(union + smooth)
       
 
        if self.reduction == 'mean':
            return IoU.mean()
        elif self.reduction == 'sum':
            return IoU.sum()
        else:
            return IoU


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))