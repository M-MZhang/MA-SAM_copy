import os
join = os.path.join
from datasets import register
import numpy as np

import pickle
import SimpleITK as sitk 
from .uitls import mkdir_if_missing, read_json, write_json


data_root = '/root/data1/zmm/seg4medicine/data/BTCV'
dataset_file = read_json()