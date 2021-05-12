import os

import torch.nn as nn
import torch
import torch.utils.data as dataL
from PIL import Image
import numpy as np
import random
from skimage.util import random_noise
from cv2 import GaussianBlur

from utils import save_img

def get_patch(lr_image, hr_image, patch_size, upscale_factor, seed):
    random.seed(seed)
    h_lr, w_lr, _ = lr_image.shape

    x_lr, y_lr = random.randint(0, w_lr-patch_size-3), random.randint(0, h_lr-patch_size-3)
    hr_patch_size, x_hr, y_hr = patch_size*upscale_factor, x_lr*upscale_factor, y_lr*upscale_factor

    lr_image = lr_image[y_lr:y_lr+patch_size, x_lr:x_lr+patch_size, :]
    hr_image = hr_image[y_hr:y_hr+hr_patch_size, x_hr:x_hr+hr_patch_size, :]

    return lr_image, hr_image



class TrainDataset(dataL.Dataset):
    def __init__(self, data_path, mode="BI"):
        super().__init__()
        self.LR_path = os.path.join(data_path, "LR_x4")
        self.HR_path = os.path.join(data_path, "HR_x4")
        self.LR_images = os.listdir(self.LR_path)
        self.HR_images = os.listdir(self.HR_path)
        self.mode = mode


    def __len__(self):
        return len(self.LR_images)

    def __getitem__(self, index):
        upscale_factor = 4
        all_patch_sizes = [60, 50, 40]
        patch_size = all_patch_sizes[upscale_factor-2]

        lr_image_path = os.path.join(self.LR_path, self.LR_images[index])
        hr_image_path = os.path.join(self.HR_path, self.HR_images[index])
        
        lr_image = Image.open(lr_image_path)
        hr_image = Image.open(hr_image_path)
        lr_image.load()
        hr_image.load()

        lr_image_data = np.asarray(lr_image, dtype=np.float32)
        hr_image_data = np.asarray(hr_image, dtype=np.float32)

        seed = 42
        lr_patch, hr_patch = get_patch(lr_image_data, hr_image_data, patch_size, upscale_factor, seed)
        
        # on testing, found several completely white patches, thus replacing them on fly
        while 35 > np.mean(lr_patch) or  np.mean(lr_patch) > 220:
            seed+=1
            lr_patch, hr_patch = get_patch(lr_image_data, hr_image_data, patch_size, upscale_factor, seed)
        
        lr_patch, hr_patch = lr_patch/255, hr_patch/255
        
        if self.mode =="DN":
            lr_patch = random_noise(lr_patch)
            
        elif self.mode == "BD":
            lr_patch = GaussianBlur(lr_patch, (7, 7), 1.6)
        
        return lr_patch, hr_patch

    def get_image_name(self, index):
        return self.LR_images[index]
