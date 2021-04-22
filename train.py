import os
from tqdm import tqdm
import logging
import datetime

import torch.nn as nn
import torch
import torch.utils.data as data
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from network import SRFBN
from data import TrainDataset


FORMAT = "%(message)s"
logs = logging.basicConfig(filename="train.log", level=logging.WARN, filemode="a", format=FORMAT)

train_data_path = os.path.join(os.getcwd(), "prepared_train")
val_data_path = os.path.join(os.getcwd(), "prepared_val")

NUM_EPOCHS = 500
BATCH_SIZE = 4
NUM_WORKERS = 24

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model_save_dir = os.path.join(os.getcwd(), "models")

model = SRFBN(scale_factor=4)
model.to(device)

criterion = nn.L1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

# get training data -> create a dataloader so that complete data is not loaded into memory
train_dataset = TrainDataset(train_data_path)
train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


write_data = []

val_dataset = TrainDataset(val_data_path)
val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    curr_data = [epoch]
    running_loss = 0.0
    last_psnr = 0
    last_ssim = 0
    dt = datetime.datetime.now()
    print(f"[{dt.hour}:{dt.minute}:{dt.second}], Running Epoch {epoch}")

    for batch_data in train_dataloader:
        # get the inputs; data is a list of [inputs, labels]
        lr_image, hr_image = batch_data
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)
        
        lr_image = torch.transpose(lr_image, 3, 1)
        hr_image = torch.transpose(hr_image, 3, 1)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(lr_image)
        curr_loss = 0.0
        for image in output:
            img_loss = criterion(image, hr_image)
            curr_loss += img_loss
        
        curr_loss/=4
        curr_loss.backward()
        running_loss += curr_loss
        
        optimizer.step()



        # progress_bar.update()
    epoch_loss = running_loss/len(train_dataloader)
    curr_data.append(epoch_loss)
    
    scheduler.step()

    # Validating
    avg_psnr = 0.0
    avg_ssim = 0.0
    with torch.no_grad(): 
        for batch_data2 in val_dataloader:
            model.eval()
            lr_image2, hr_image2 = batch_data2
            hr_image2 = hr_image2.squeeze()

            lr_image2 = lr_image2.to(device)
            lr_image2 = torch.transpose(lr_image2, 3, 1)
            outputs_images2 = model(lr_image2)

            sr_image = outputs_images2[-1].T.squeeze()
            sr_image = sr_image.detach().cpu().numpy()
            hr_image2 = hr_image2.detach().cpu().numpy()
            
            img_psnr = psnr(hr_image2, sr_image)
            img_ssim = ssim(hr_image2, sr_image, multichannel=True)
            avg_psnr += img_psnr
            avg_ssim += img_ssim
    
    avg_psnr/=len(val_dataloader)
    avg_ssim/=len(val_dataloader)
    curr_data.append(avg_psnr)
    curr_data.append(avg_ssim)
    write_data.append(curr_data)
    logging.warn(f"{epoch},{epoch_loss},{avg_psnr},{avg_ssim}")

    if (avg_psnr > last_psnr) or (avg_ssim > last_ssim):
        if last_psnr > avg_psnr:
            last_psnr = avg_psnr
        if last_ssim > avg_ssim:
            last_ssim = avg_ssim
        
        filename = os.path.join(model_save_dir, f'Epoch_{epoch}_ckpt.pth')
        ckp = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': epoch_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
        torch.save(ckp, filename)
        logging.warn("SAVING MODEL")


import csv
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(curr_data)