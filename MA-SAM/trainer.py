import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss, MultiClassFocalLoss, BinaryDiceLoss
from torchvision import transforms
from icecream import ic
from datetime import datetime
from test import inference, inference_2d
from torch.optim.lr_scheduler import _LRScheduler



# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

def calc_loss(outputs, low_res_label_batch, ce_loss, focal_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    # loss_focal = focal_loss(low_res_logits, low_res_label_batch)
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)

class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


def trainer_run(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset import dataset_reader, RandomGenerator
    
   
    
    if not os.path.exists(args.output + '/training_log'): # 换到外面去存储
        os.mkdir(args.output + '/training_log')
    # time
    output_filename = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logging.basicConfig(filename= args.output + '/training_log/' + args.output.split('/')[-1] + '_'+ output_filename + '_log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes 
    batch_size = args.batch_size * args.n_gpu
    
    db_train = dataset_reader(base_dir=args.root_path, split="train", num_classes=args.num_classes, 
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=False) # 这个drop_last好像会有点什么问题？
    
    num = 0
    for name, para in model.named_parameters():
        if 'image_encoder' not in name:
            para.requires_grad_(False)
            if "task_specific_embed_list" in name: #这一步就已经将mask_decoder中的mask_tokens的梯度置为true了
                para.requires_grad_(True)
                num += para.numel()
                # print(name)
            elif "Neck_list" in name:
                para.requires_grad_(True)
                num += para.numel()
            elif "task_adapter" in name:
                para.requires_grad_(True)
                num += para.numel()
            elif "mask_decoder" in name and 'sam' not in name:
                para.requires_grad_(True)
                num += para.numel()
    
    # varify the trainable parameters
    for name, para in model.named_parameters():
        if para.requires_grad:
            print(name)
            logging.info(name)
    logging.info("The number of trainable parameters is {}M".format(num/1000000))

    # model.init_weights() # 将加入到image_encoder中的adapter_mlp层最后一层的参数初始化为0

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss(ignore_index=-100)
    focal_loss = Focal_loss(alpha=0.1, num_classes=num_classes + 1, gamma=5)
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001) 
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, float(args.max_epochs)
    #     )
    
    # if args.warmup:
    #     scheduler = ConstantWarmupScheduler(
    #             optimizer, scheduler, args.warmup_period,
    #             1e-5
    #         )
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    iterator = tqdm(range(max_epoch), ncols=70)

    # 测试最基础的版本
    inference_2d(args, multimask_output, model,  low_res, None)

    best_dice = -np.inf
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
            hw_size = image_batch.shape[-1]
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)

            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            
            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size)
                    loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, focal_loss, dice_loss, args.dice_param)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                # lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            # lr = scheduler.get_last_lr()
            iter_num = iter_num + 1
            writer.add_scalar('info/lr',lr_ , iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(),lr_))

        save_interval = 10
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)
                # torch.save(model.module.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            dice = inference_2d(args, multimask_output, model,  low_res, None)
            if dice > best_dice:
                best_dice = dice
                save_mode_path = os.path.join(snapshot_path, 'best.pth')

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)
                # torch.save(model.module.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
