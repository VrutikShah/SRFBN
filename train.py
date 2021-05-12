import os
import logging
import datetime

import torch.nn as nn
import torch
import torch.utils.data as data
import torch.optim as optim

from skimage.util import random_noise
from cv2 import GaussianBlur

from network import SRFBN
from data import TrainDataset
from utils import init_weights, get_psnr_ssim


# Pre-defined Constants and PATHS
train_data_path = os.path.join(os.getcwd(), "train_data")
val_data_path = os.path.join(os.getcwd(), "val_data")

NUM_EPOCHS = 500

# adjust the batch size depending upon GPU memory available. Batch size of 16 takes around 9Gb of mem.
BATCH_SIZE = 16
VAL_BATCH_SIZE = 8

# if not on WINDOWS, can increase the value below to 4 or 6. 
NUM_WORKERS=1
T = 4
GPU_ID = 0
MODE = "BI"
LOAD_PRE = False
NAME_PRETRAINED = ""


device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
model_save_dir = os.path.join(os.getcwd(), "models")


# defining the model and its parameters, to run inception model, pass fe="inception" as and additional parameter
model = SRFBN(scale_factor=3, T=T)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

if LOAD_PRE:
    model_path = os.path.join(model_save_dir, NAME_PRETRAINED)
    chkpt = torch.load(model_path)
    model.load_state_dict(chkpt['state_dict'])
    model = nn.DataParallel(model, device_ids=[GPU_ID])
    optimizer.load_state_dict(chkpt['optimizer'])
else:
    model = nn.DataParallel(model, device_ids=[GPU_ID])
    model.apply(init_weights)


# L1 Loss
criterion = nn.L1Loss()

# Learning Rate Scheduler, halves learning rate after 200 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
model.to(device)


# get training data -> create a dataloader so that complete data is not loaded into memory
train_dataset = TrainDataset(train_data_path, mode=MODE)
train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


write_data = []

val_dataset = TrainDataset(val_data_path)
val_dataloader = data.DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

last_psnr = 0
last_ssim = 0

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    curr_data = [epoch]
    running_loss = 0.0
    
    dt = datetime.datetime.now()
    print(f"[{dt.hour}:{dt.minute}:{dt.second}], Running Epoch {epoch}")

    for batch_data in (train_dataloader):
        # get the inputs; data is a list of [lr_image, hr_image]
        lr_image, hr_image = batch_data
        lr_image = lr_image.float().to(device)
        
        if MODE=="BD":
            blur_HR = GaussianBlur(hr_image.cpu().numpy(), (7, 7), 1.6)
            blur_HR = torch.from_numpy(blur_HR).float().to(device)
            blur_HR = torch.transpose(blur_HR, 3, 1)
        elif MODE=="DN":
            noisy_HR = random_noise(hr_image)
            noisy_HR = torch.from_numpy(noisy_HR).float().to(device)
            noisy_HR = torch.transpose(noisy_HR, 3, 1)
        
        hr_image = hr_image.to(device)

        lr_image = torch.transpose(lr_image, 3, 1)
        hr_image = torch.transpose(hr_image, 3, 1)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(lr_image)
        curr_loss = 0.0
        for idx, image in enumerate(output):

            # compare with noisy/blurred image
            if idx<2:
                if MODE=="BI":
                    img_loss = criterion(image, hr_image)
                elif MODE=="BD":
                    img_loss = criterion(image, blur_HR)
                elif MODE=="DN":
                    img_loss = criterion(image, noisy_HR)

            else:
                img_loss = criterion(image, hr_image)
            curr_loss += img_loss
        
        curr_loss/=T
        curr_loss.backward()
        running_loss += curr_loss
        
        optimizer.step()
    
    
    epoch_loss = running_loss/len(train_dataloader)
    curr_data.append(epoch_loss)
    
    scheduler.step()

    # Validating
    avg_psnr = 0.0
    avg_ssim = 0.0
    model.eval()
    with torch.no_grad():
        for batch_data2 in (val_dataloader):
            lr_image2, hr_image2 = batch_data2
            hr_image2 = hr_image2.squeeze()

            lr_image2 = lr_image2.to(device)
            lr_image2 = torch.transpose(lr_image2, 3, 1)
            outputs_images2 = model(lr_image2)

            sr_image = torch.transpose(outputs_images2[-1], 1, 3)
            sr_image = sr_image.detach().cpu().numpy()
            hr_image2 = hr_image2.detach().cpu().numpy()
            for idx in range(hr_image2.shape[0]):
                img_psnr, img_ssim = get_psnr_ssim(sr_image[idx, :, :, :], hr_image2[idx, :, :, :])
                avg_psnr += img_psnr
                avg_ssim += img_ssim            
    
    avg_psnr/=len(val_dataset)
    avg_ssim/=len(val_dataset)
    curr_data.append(avg_psnr)
    curr_data.append(avg_ssim)
    write_data.append(curr_data)
    logging.warn(f"{epoch},{epoch_loss},{avg_psnr},{avg_ssim}")

    # saving the model if higher value is recieved for either of the metrics
    if (avg_psnr > last_psnr) or (avg_ssim > last_ssim):
        last_psnr = max(last_psnr, avg_psnr)
        last_ssim = max(last_ssim, avg_ssim)
        
        filename = os.path.join(model_save_dir, f'B_incep_Epoch_{epoch}_ckpt.pt')
        ckp = {
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': epoch_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
        torch.save(ckp, filename)
        logging.warn("SAVING MODEL")


# in the end, savinf all the loss to a csv file.
import csv
with open("results_B_incep_final.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(write_data)