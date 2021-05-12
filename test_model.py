import os

import torch.nn as nn
import torch
import torch.utils.data as data
import torch.optim as optim

from network import SRFBN
from data import TrainDataset
from utils import save_img, get_psnr_ssim

BATCH_SIZE = 1
MODEL_NAME = "DN_Epoch_71_ckpt.pt"
SAVE_IMAGES = False
FE_BLOCK = "normal" # or normal


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.getcwd(),"models", MODEL_NAME)

model = SRFBN(scale_factor=4, fe_block=FE_BLOCK)
chkpt = torch.load(model_path)
model.load_state_dict(chkpt['state_dict'])
model.to(device)

all_test_dataset_path = os.path.join(os.getcwd(), "test_data")
all_test_dataset = os.listdir(all_test_dataset_path)

# iterating over datasets
for dataset in all_test_dataset:
    print(f'Testing on {dataset}')
    test_dataset_path = os.path.join(all_test_dataset_path, dataset)

    test_dataset = TrainDataset(test_dataset_path)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    avg_psnr = 0.0
    avg_ssim = 0.0
    max_psnr = 0.0
    max_ssim = 0.0
    img_name1 = ""
    img_name2 = ""
    with torch.no_grad():
        for idx, batch_data in enumerate(test_dataloader):
            lr_image, hr_image = batch_data

            lr_image = lr_image.to(device)
            lr_image = torch.transpose(lr_image, 3, 1)
            outputs_images2 = model(lr_image)

            sr_image = outputs_images2[-1]
            sr_image = torch.transpose(sr_image, 1, 3)

            lr_image = lr_image.transpose(1, 3)

            lr_image = lr_image.detach().cpu().numpy()
            sr_image = sr_image.detach().cpu().numpy()
            hr_image = hr_image.detach().cpu().numpy()
            for idx2 in range(sr_image.shape[0]):
                img_psnr, img_ssim = get_psnr_ssim(sr_image[idx2, :, :, :], hr_image[idx2, :, :, :])
                if img_psnr > max_psnr:
                    max_psnr = img_psnr
                    img_name1 = test_dataset.get_image_name(BATCH_SIZE*idx+idx2)
                    if SAVE_IMAGES:
                        rc = save_img(sr_image[idx2, :, :, :], img_name1)
                        if rc==-1:
                            continue
                        save_img(hr_image[idx2, :, :, :], img_name1, "HR")
                        save_img(lr_image[idx2, :, :, :], img_name1, "LR")

                    
                if img_ssim > max_ssim:
                    max_ssim = img_ssim
                    if SAVE_IMAGES:
                        img_name2 = test_dataset.get_image_name(BATCH_SIZE*idx+idx2)
                        rc = save_img(sr_image[idx2, :, :, :], img_name2)
                        if rc==-1:
                            continue
                        save_img(hr_image[idx2, :, :, :], img_name2, "HR")
                        save_img(lr_image[idx2, :, :, :], img_name2, "LR")

                avg_psnr += img_psnr
                avg_ssim += img_ssim

    avg_psnr/=len(test_dataset)
    avg_ssim/=len(test_dataset)
    print(f"{dataset},{avg_psnr},{avg_ssim},{max_psnr},{max_ssim},{img_name1},{img_name2}")
