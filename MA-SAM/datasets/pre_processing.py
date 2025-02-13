import os
join = os.path.join
import numpy as np

import pickle
import SimpleITK as sitk 
import json
import errno
import os.path as osp
import warnings


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile

def check_isdir(dpath):
    """Check if the given path is a directory.

    Args:
        dpath (str): directory path.

    Returns:
       bool
    """
    isdir = osp.isdir(dpath)
    if not isdir:
        warnings.warn('No directory found at "{}"'.format(dpath))
    return isdir

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))

data_root = '/root/data1/zmm/seg4medicine/data/BTCV/'
dataset_file = read_json(data_root+'dataset_0.json')

surfix = '.nii.gz'

train_file_list = dataset_file['training']
val_file_list = dataset_file['validation']

train_list = []
for train_file in train_file_list:
    train_image_path = data_root + train_file['image']
    train_seg_path = data_root + train_file['label']
    id = train_file['image'].split("/")[1].split(surfix)[0]
    save_path = data_root + 'npy_new/' + id
    mkdir_if_missing(save_path+'/images')
    mkdir_if_missing(save_path+'/masks')

    image = sitk.ReadImage(train_image_path)
    masks = sitk.ReadImage(train_seg_path)

    image_arr = sitk.GetArrayFromImage(image)
    masks_arr = sitk.GetArrayFromImage(masks)

    for slice in range(image_arr.shape[0]):
        image_slice_3c = np.repeat(image_arr[slice][:, :, None], 3, axis=-1)
        image_slice_save_path = join(save_path, 'images', str(slice).zfill(3) + '.pkl')
        seg_slice_save_path = image_slice_save_path.replace("images", "masks")

        with open(image_slice_save_path, "wb") as f:
            pickle.dump(image_slice_3c, f)
                    
        with open(seg_slice_save_path, "wb") as f:
            pickle.dump(masks_arr[slice], f)
        
        train_list.append({'images':image_slice_save_path, 'masks':seg_slice_save_path})

val_list = []
for val_file in val_file_list:
    val_image_path = data_root + val_file['image']
    val_seg_path = data_root + val_file['label']
    id = val_file['image'].split("/")[1].split(surfix)[0]
    save_path = data_root + 'npy_new/' + id
    mkdir_if_missing(save_path+'/images')
    mkdir_if_missing(save_path+'/masks')

    image = sitk.ReadImage(val_image_path)
    masks = sitk.ReadImage(val_seg_path)

    image_arr = sitk.GetArrayFromImage(image)
    masks_arr = sitk.GetArrayFromImage(masks)

    for slice in range(image_arr.shape[0]):
        image_slice_3c = np.repeat(image_arr[slice][:, :, None], 3, axis=-1)
        image_slice_save_path = join(save_path, 'images', str(slice).zfill(3) + '.pkl')
        seg_slice_save_path = image_slice_save_path.replace("images", "masks")

        with open(image_slice_save_path, "wb") as f:
            pickle.dump(image_slice_3c, f)
                    
        with open(seg_slice_save_path, "wb") as f:
            pickle.dump(masks_arr[slice], f)
        
        val_list.append({'images':image_slice_save_path,'masks':seg_slice_save_path})

split = {'train':train_list, 'val':val_list}
write_json(split, data_root+'npy_new.json')
print(f"Finish Saving split to {data_root}")