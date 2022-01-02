import numpy as np 
import pdb 
from PIL import Image
import os 
import pdb 
import torch 
import os 
import scipy.io as sio
import pdb 

dirs = '/root/dataset2/PASCAL3D+_release1.1/Annotations/car_pascal'
dir_path = os.listdir(dirs)

for path in dir_path:
    anno_path = os.path.join(dirs, path)
    annos = sio.loadmat(anno_path)['record']        # mat 하나하나마다 불러옴 
    # trans = CameraTransformer(annos).get_transformation_matrix()
    pdb.set_trace()