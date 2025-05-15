import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from unetplusplus import UnetPlusPlus

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
from utils import BinaryDiceLoss,IoU
from torchvision import transforms
from icecream import ic
from datetime import datetime
from utils import calculate_metric_percase, write_json, HD_Score
from datasets.dataset import dataset_reader, RandomGenerator, test_transform
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    out = outputs.squeeze(1)
    loss_dice = dice_loss(out, low_res_label_batch)
    loss_ce = ce_loss(out, low_res_label_batch.float())
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def inference_2d(args, model,logger, test_save_path=None):
    hd_score = HD_Score(n_classes=args.num_classes)
    iou_score = IoU()
    model.eval()
    low_res = args.img_size // 4
    db_test = dataset_reader(base_dir=args.data_path, split="test", num_classes=args.num_classes, 
                            transform=transforms.Compose([test_transform(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]),
                            test_name=None)

    print("The length of test set is: {}".format(len(db_test)))
    
    batch_size = args.batch_size * args.n_gpu
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    testdataloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            worker_init_fn=worker_init_fn, drop_last=False)

    iou = 0
    dice = 0
    num_test = 0
    h_num = 0
    for i_batch, sampled_batch in enumerate(testdataloader):
       
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        hw_size = image_batch.shape[-1]
        label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        
        with torch.no_grad():
            outputs = model(image_batch)
            outputs = outputs.squeeze(1)
            iou += iou_score(outputs, label_batch) * label_batch.shape[0]
         
            # h = hd_score(outputs, label_batch)
            # if h != float("inf") and np.isnan(h) == False:
            #     hd.append(h)
            
            out = outputs.cpu().detach().numpy()
            label_batch = label_batch.cpu().detach().numpy()
            out[out<=0] = 0
            dice += calculate_metric_percase(out, label_batch) * label_batch.shape[0]
            num_test += image_batch.shape[0]

            #可视化一下
            # if i_batch <10 :
            #     img = Image.fromarray(np.array(out[0]*255).squeeze().astype(np.uint8))
            #     img.save(os.path.join(args.visual_path, str(i_batch)+'.png'))
            #     label = Image.fromarray(np.array(label_batch[0]*255).squeeze().astype(np.uint8))
            #     label.save(os.path.join(args.visual_path,str(i_batch)+'_label.png'))

            # save mask
            img = Image.fromarray(np.array(out[0]*255).squeeze().astype(np.uint8))
            img.save(os.path.join(args.visual_path, str(i_batch)+'_mask.png'))
            
             
    iou = iou / num_test
    dice = dice / num_test

    logger.info("DICE:{}, IoU:{}".format(dice, iou))
    
    loss = {'DICE':dice, 'IoU': iou}
    if test_save_path is not None:
        write_json(loss, test_save_path+'/result.json')
    print("Finish test haha!")
    return dice


def trainer_run(args, model, snapshot_path):
    if not os.path.exists(args.output + '/training_log'): # 换到外面去存储
        os.mkdir(args.output + '/training_log')
    # time
    output_filename = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    # 2. 创建文件处理器
    file_handler = logging.FileHandler(filename= args.output + '/training_log/' + args.output.split('/')[-1] + '_'+ output_filename + '_log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes 
    batch_size = args.batch_size * args.n_gpu
    
    db_train = dataset_reader(base_dir=args.root_path, split="train", num_classes=args.num_classes, 
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[args.img_size, args.img_size])]),)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=False) # 这个drop_last好像会有点什么问题？
    
    num = 0
    for name, para in model.named_parameters():
        para.requires_grad = True
        num += para.numel()
    logger.info("The number of parameters is {}M".format(num/1000000))
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = nn.BCEWithLogitsLoss()
    # dice_loss = DiceLoss(num_classes+1)
    dice_loss = BinaryDiceLoss()
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001) 
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

   
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    iterator = tqdm(range(max_epoch), ncols=70)

    # 测试最基础的版本
    best_dice = inference_2d(args, model, logger, None)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
            hw_size = image_batch.shape[-1]
            
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
    
            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch)
                    loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss, args.dice_param)
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
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr',lr_ , iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logger.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(),lr_))

        save_interval = 10
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                # model.save_parameters(save_mode_path)
                torch.save(model.state_dict(), save_mode_path)
            except:
                # model.module.save_parameters(save_mode_path)
                torch.save(model.module.state_dict(), save_mode_path)
            logger.info("save model to {}".format(save_mode_path))
            dice = inference_2d(args, model,  logger, None)
            if dice > best_dice:
                best_dice = dice
                save_mode_path = os.path.join(snapshot_path, 'best.pth')
                try:
                    # model.save_parameters(save_mode_path)
                    torch.save(model.state_dict(), save_mode_path)
                except:
                    # model.module.save_parameters(save_mode_path)
                    torch.save(model.module.state_dict(), save_mode_path)
                logger.info("save best model {} to {}".format('epoch_' + str(epoch_num) , save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)
            logger.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/data1/zmm/seg4medicine/data/isic2017', help='root dir for data')
parser.add_argument('--output', type=str, default='/root/data1/zmm/seg4medicine/save/unet++/isic2017', help='output dir for model and log')
parser.add_argument('--visual_path', type=str, default='/root/data1/zmm/seg4medicine/visualization/unet++/isic2017', help='visualization dir for model and log')
parser.add_argument('--data_path', type=str, default='/root/data1/zmm/seg4medicine/data/isic2017', help='data dir')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--base_lr', type=float, default=0.0002, help='segmentation network learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')

parser.add_argument('--max_epochs', type=int,default=400, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int, default=300, help='maximum epoch number to train')

parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_name', type=str, default='vit_h', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='/root/data1/zmm/seg4medicine/pretrained/sam_vit_h_4b8939.pth', help='Pretrained checkpoint')
parser.add_argument('--adapt_ckpt', type=str, default='/root/data1/zmm/seg4medicine/save/unet++/isic2017/epoch_169.pth', help='Finetuned checkpoint')
parser.add_argument('--rank', type=int, default=32, help='Rank for FacT')
parser.add_argument('--scale', type=float, default=1.0, help='Scale for FacT')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid when warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='task_specific_sam')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--lr_exp', type=float, default=2, help='The learning rate decay expotential')

# acceleration choices
parser.add_argument('--tf32', action='store_true', help='If activated, use tf32 to accelerate the training process')
parser.add_argument('--compile', action='store_true', help='If activated, compile the training model for acceleration')
parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration')
parser.add_argument('--skip_hard', action='store_true', help='If activated, adopt mixed precision for acceleration')
parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')

args = parser.parse_args()
args.warmup = True
args.AdamW = True
args.tf32 = True
args.compile = False
args.use_amp = True
args.skip_hard = True

if __name__ == "__main__":
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    if not os.path.exists(args.visual_path):
        os.makedirs(args.visual_path)

    # register model
    
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False


    config_file = os.path.join(args.output, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)
    
    model = UnetPlusPlus(num_classes=args.num_classes)
    
    # test 
    # trainer_run(args, model, args.output)
    if args.adapt_ckpt is not None:
        print("Load the adapted checkpoint from {}".format(args.adapt_ckpt))
        state_dict = torch.load(args.adapt_ckpt, map_location='cpu')
        for key in list(state_dict.keys()):
            if key.startswith('module.'):
                state_dict[key[7:]] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    
    model = model.cuda()

    # initialize log_folder
    log_folder = os.path.join(args.output, 'testing_log')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(args.visual_path):
        os.makedirs(args.visual_path)
     # time
    output_filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    # 2. 创建文件处理器
    file_handler = logging.FileHandler(filename=log_folder+'/'+args.adapt_ckpt.split('/')[-1] +'_log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    

    logger.info(str(args))
    
    if args.is_savenii:
        test_save_path = log_folder
    else:
        test_save_path = None
   
    inference_2d(args, model, logger, test_save_path)

  
