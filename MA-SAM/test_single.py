import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry
from torch.nn.modules.loss import CrossEntropyLoss
from PIL import Image
import cv2

from icecream import ic
import pandas as pd
import pickle
from datetime import datetime
from einops import repeat
from scipy.ndimage import zoom
from utils import calculate_metric_percase
import torch.nn.functional as F
# import nibabel as nib

# from datasets.dataset import dataset_reader, RandomGenerator, test_transform
# from torchvision import transforms
# import json
# from torch import nn

# from mindspore.nn.metrics import HausdorffDistance

HU_min, HU_max = -200, 250
data_mean = 50.21997497685108
data_std = 68.47153712416372



def test_single_volume(image, label, net, classes, multimask_output, patch_size=[512, 512], test_save_path=None, case=None):
    
    image, label = image.squeeze(0), label.squeeze(0) #[d, h, w, 3], [d, h, w]
    # label = label[:,:,:,2]
    
    probability = np.expand_dims(np.zeros_like(label, dtype=np.float32), axis=-1) #[d, h, w, 1]
    probability = repeat(probability, 'd h w c -> d h w (repeat c)', repeat=classes+1) #[d, h, w, classes+1]

    # probability = np.concatenate((probability[0:1], probability[0:1], probability, probability[-1:], probability[-1:]), axis=0)

    avg_cnt = np.ones_like(probability, dtype=np.float32) #[d, h, w, classes+1]
    for ind in range(image.shape[0]):
        slice = image[ind]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
        
        inputs = torch.from_numpy(slice).unsqueeze(0).float().cuda() #[b, h, w, c]
        # inputs = repeat(inputs, 'b h w c -> b c h w', repeat=3)
        inputs = torch.permute(inputs, (0, 3, 1, 2))
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1)
            out = out.cpu().detach().numpy()
            out_pred = torch.softmax(output_masks, dim=1)
            out_pred = torch.permute(out_pred, (0, 2, 3, 1))
            out_pred = out_pred.cpu().detach().numpy()
            out_h, out_w = out.shape[1], out.shape[2]
            if x != out_h or y != out_w:
                out_pred = zoom(out_pred, (1.0, x / out_h, y / out_w, 1.0), order=3)
            
            probability[ind] += out_pred[0]
            avg_cnt[ind] += 1.
            
    probability = probability/avg_cnt
    prediction = np.argmax(probability, axis=-1)
    # prediction = prediction[2:-2]

    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        
        image_data = np.moveaxis(image[:,:,:,2].astype(np.float32), 0, -1)
        prediction_data = np.moveaxis(prediction.astype(np.float32), 0, -1)
        label_data = np.moveaxis(label.astype(np.float32), 0, -1)

        image_data = np.rot90(np.flip(image_data, axis=1), k=-1, axes=(0, 1))
        prediction_data = np.rot90(np.flip(prediction_data, axis=1), k=-1, axes=(0, 1))
        label_data = np.rot90(np.flip(label_data, axis=1), k=-1, axes=(0, 1))

        # Create Nifti images
        # img_nifti = nib.Nifti1Image(image_data, np.eye(4))
        # prd_nifti = nib.Nifti1Image(prediction_data, np.eye(4))
        # lab_nifti = nib.Nifti1Image(label_data, np.eye(4))

        # # Set spacing
        # img_nifti.header['pixdim'][1:4] = [1, 1, 1]
        # prd_nifti.header['pixdim'][1:4] = [1, 1, 1]
        # lab_nifti.header['pixdim'][1:4] = [1, 1, 1]

        # # Save the images
        # img_nifti.to_filename(f"{test_save_path}/{case}_img.nii.gz")
        # prd_nifti.to_filename(f"{test_save_path}/{case}_pred.nii.gz")
        # lab_nifti.to_filename(f"{test_save_path}/{case}_gt.nii.gz")
        
    return metric_list



def inference_single(args, multimask_output, model, test_save_path=None):

    model.eval()
    image_name = 'karvis.png'
    image_path = os.path.join(args.visual_path,image_name)
    mask_path = os.path.join(args.visual_path, image_name.split('.')[0] + '-gt.png')
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # preprocessing
    # Normalization
    image = ((image-np.min(image))/(np.max(image)-np.min(image)+0.00000001))

    x, y, z = image.shape
    output_size = (args.img_size, args.img_size)
    if x!=output_size or y!=output_size:
        image = zoom(image, (output_size[0] / x, output_size[1] / y, 1.0), order=3)

    image = torch.from_numpy(image.astype(np.float32))
    image = image.permute(2,0,1)

    # unsqueeze for batch

    image_batch = image.unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(image_batch, multimask_output, args.img_size)

        low_res_logits = outputs['low_res_logits']
        out = torch.argmax(torch.softmax(low_res_logits, dim=1), dim=1)
        out = out.cpu().detach().numpy()
        out = zoom(out, (1.0, x / out.shape[1], y / out.shape[2]), order=3)
    
    # save mask
    img = Image.fromarray(np.array(out[0]*255).squeeze().astype(np.uint8))
    img.save(os.path.join(args.visual_path, image_name.split('.')[0] +'_mask.png'))
            
    # 存储热力图需要的embedding
    # attn = {'encoder': encoder_attns, 'decoder': decoder_attns}
    # torch.save(attn, os.path.join(args.visual_path, image_name.split('.')[0] + '_attn.pt'))

    print("Finish test haha!")

def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapt_ckpt', type=str, default='/root/autodl-tmp/save/v6.7_polyp_H_3/epoch_159.pth', help='The checkpoint after adaptation')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/visualization/Appendix/segmentation/Polyp', help='The path of the dataset')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/visualization/Appendix/segmentation/Polyp')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu') 
    parser.add_argument('--visual_path', type=str, default='/root/autodl-tmp/visualization/Appendix/segmentation/Polyp')  
    
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='/root/autodl-tmp/pretrained/sam_vit_h_4b8939.pth', help='Pretrained checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_h', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=32, help='Rank for FacT adaptation')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--module', type=str, default='task_specific_sam')

    args = parser.parse_args()

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
    

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0., 0., 0.],
                                                                pixel_std=[1., 1., 1.])
    
    pkg = import_module(args.module)
    net = pkg.Sam_task(sam, r=32).cuda() 
    # net = sam.cuda()

    assert args.adapt_ckpt is not None
    net.load_parameters(args.adapt_ckpt)
    # net.load_state_dict(torch.load(args.adapt_ckpt))
   

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log_folder
    log_folder = os.path.join(args.output_dir, 'testing_log')
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

    low_res = img_embedding_size * 4
   
   
    # _ = inference_2d(args, multimask_output, net,  low_res, logger, log_folder)
    inference_single(args, multimask_output, net, test_save_path)

