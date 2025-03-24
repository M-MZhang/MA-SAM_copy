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
from utils import DiceLoss

from icecream import ic
import pandas as pd
import pickle
from datetime import datetime
from einops import repeat
from scipy.ndimage import zoom
from utils import calculate_metric_percase, write_json, IoU
import nibabel as nib

from datasets.dataset import dataset_reader, RandomGenerator
from torchvision import transforms
import json

HU_min, HU_max = -200, 250
data_mean = 50.21997497685108
data_std = 68.47153712416372

# os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1.0'

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
        img_nifti = nib.Nifti1Image(image_data, np.eye(4))
        prd_nifti = nib.Nifti1Image(prediction_data, np.eye(4))
        lab_nifti = nib.Nifti1Image(label_data, np.eye(4))

        # Set spacing
        img_nifti.header['pixdim'][1:4] = [1, 1, 1]
        prd_nifti.header['pixdim'][1:4] = [1, 1, 1]
        lab_nifti.header['pixdim'][1:4] = [1, 1, 1]

        # Save the images
        img_nifti.to_filename(f"{test_save_path}/{case}_img.nii.gz")
        prd_nifti.to_filename(f"{test_save_path}/{case}_pred.nii.gz")
        lab_nifti.to_filename(f"{test_save_path}/{case}_gt.nii.gz")
        
    return metric_list

def inference(args, multimask_output, model, test_save_path=None):
    data_fd_list = ['0035', '0036', '0037', '0038', '0039', '0040']
    
    model.eval()
    metric_list = []
    for data_fd in tqdm(data_fd_list):
        image_file_path = args.data_path +'/npy_new'+'/img'+str(data_fd)+'/images'
        image_file_list = os.listdir(image_file_path)
        image_file_list.sort()
        image_arr_list = []
        mask_arr_list = []
        for image_file in image_file_list:
            with open(image_file_path + '/' + image_file, 'rb') as file:
                image_arr = pickle.load(file)
            with open(args.data_path + '/npy_new'+'/img'+str(data_fd)+'/masks/'+image_file, 'rb') as file:
                mask_arr = pickle.load(file)

            image_arr = np.clip(image_arr, HU_min, HU_max)
            image_arr = (image_arr-HU_min)/(HU_max-HU_min)*255.0
            image_arr = np.float32(image_arr)
            image_arr = (image_arr - data_mean) / data_std
            image_arr = (image_arr-image_arr.min())/(image_arr.max()-image_arr.min()+0.00000001)

            mask_arr = np.float32(mask_arr)
            if args.num_classes==12:
                mask_arr[mask_arr==13] = 12
                class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5:'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta', 9:'vena', 10:'vein', 11: 'pancreas', 12:'adrenal gland'}
            
            image_arr_list.append(image_arr)
            mask_arr_list.append(mask_arr)

        image = np.expand_dims(np.stack(image_arr_list), axis=0) #[1, d, h, w, 3]
        label = np.expand_dims(np.stack(mask_arr_list), axis=0) #[1, d, h, w]
        case_name = data_fd

        h, w = image.shape[2], image.shape[3]

        metric_i = test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                        patch_size=[args.img_size, args.img_size],
                                        test_save_path=test_save_path, case=case_name)
        
        metric_list.append(np.array(metric_i))
        logging.info('idx %d case %s mean_dice %f' % (
            1, case_name, np.nanmean(metric_i, axis=0)))
    
    metric_list = np.nanmean(metric_list, axis=0)
    for i in range(1, args.num_classes + 1):
        logging.info('Mean class %d name %s mean_dice %f' % (i, class_to_name[i], metric_list[i - 1]))

    performance = np.nanmean(metric_list, axis=0)
    
    logging.info('Testing performance in best val model: mean_dice : %f ' % (performance))
    logging.info("Testing Finished!")
    return 1

def inference_2d(args, multimask_output, model, low_res, test_save_path=None):
    Polyp_name = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    # Polyp_name = ['CVC-300']
    iou_loss = IoU(reduction='mean')
  
    iou_dict = {}
    dice_dict = {}
    model.eval()
    for polyp in Polyp_name:
        db_test = dataset_reader(base_dir=args.data_path, split="test", num_classes=args.num_classes, 
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]),
                                test_name=polyp)
    
        print("The length of test set is: {}".format(len(db_test)))
        
        batch_size = args.batch_size * args.n_gpu
        def worker_init_fn(worker_id):
            random.seed(args.seed + worker_id)

        testdataloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                worker_init_fn=worker_init_fn, drop_last=False)

        iou = 0
        dice = 0
        num_test = 0
        for i_batch, sampled_batch in enumerate(testdataloader):
            # print(i_batch)
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
            hw_size = image_batch.shape[-1]
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            with torch.no_grad():
                outputs = model(image_batch, multimask_output, args.img_size)
                low_res_logits = outputs['low_res_logits']
                
                out = torch.argmax(torch.softmax(low_res_logits, dim=1), dim=1)
                out = out.cpu().detach().numpy()
                label_batch = label_batch.cpu().detach().numpy()
                iou += iou_loss(out, label_batch) * image_batch.shape[0]
                dice += calculate_metric_percase(out, label_batch) * image_batch.shape[0]

                num_test += image_batch.shape[0]
        

    iou = iou / num_test
    dice = dice / num_test
    
        # iou_dict[polyp] = iou
        # dice_dict[polyp] = dice
        # print("{}: DICE:{}, IoU:{}".format(polyp, dice, iou))
    # print("DICE:{}, IoU:{}".format(dice, iou))s
    logging.info("DICE:{}, IoU:{}".format(dice, iou))
    
    loss = {'DICE':dice, 'IoU': iou}
    if test_save_path is not None:
        write_json(loss, test_save_path+'/result.json')
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
    parser.add_argument('--adapt_ckpt', type=str, default='/root/autodl-tmp/save/v6.7_isic2018_B_2/epoch_79.pth', help='The checkpoint after adaptation')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/isic2018')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/save/v6.7_isic2018_B_2')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')   
    
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='/root/autodl-tmp/pretrained/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
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
   

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log_folder
    log_folder = os.path.join(args.output_dir, 'testing_log')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # time
    output_filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logging.basicConfig(filename= log_folder+args.adapt_ckpt.split('/')[-1] +'_log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    if args.is_savenii:
        test_save_path = log_folder
    else:
        test_save_path = None


    low_res = img_embedding_size * 4
    inference_2d(args, multimask_output, net,  low_res, log_folder)

