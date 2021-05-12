import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import numpy as np
import os
from PIL import Image
import cv2

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# initializing weights as given in the paper.
def init_weights(m):
    if type(m) == nn.Conv2d:
        kaiming_normal_(m.weight, mode="fan_in")

    elif type(m) == nn.BatchNorm2d:
        constant_(m.weight, 1.)
        constant_(m.bias, 0.)


# saving images to view results
def save_img(output_patch, img_name, type_="SR"):
    output_patch *=255
    output_patch = output_patch.clip(0, 255)

    # not saving img that are largely white/black
    if np.mean(output_patch) > 240 or np.mean(output_patch) < 15:
        return -1

    output_patch = np.uint8(output_patch)
    output_patch= Image.fromarray(output_patch)
    
    if not os.path.exists("best_test_images"):
        os.mkdir("best_test_images")
    
    save_name2 = os.path.join(os.getcwd(), "best_test_images", f"{img_name}_{type_}.png")
    output_patch.save(save_name2)
    return 0

def get_psnr_ssim(pred, gt):
    # evaluating the psnr and ssim on Y channel of the image.
    pred1 = cv2.cvtColor(pred, cv2.COLOR_RGB2YUV)
    y_pred, _, _ = cv2.split(pred1)

    gt1 = cv2.cvtColor(gt, cv2.COLOR_RGB2YUV)
    y_gt, _, _ = cv2.split(gt1)

    curr_psnr = psnr(y_gt, y_pred)
    curr_ssim = ssim(y_gt, y_pred)

    return curr_psnr, curr_ssim


# print number of parameters in the model
def get_number_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)