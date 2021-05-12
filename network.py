import torch
import torch.nn as nn
import logging

#? HYPER-PARAMETERS
#? G=6, m=32, T=4 (SRFBN-L)
#! PReLu layer after all conv and deconv blocks except the last one
#! Batch Norm between conv layer and prelu layer


logging.basicConfig(filename="network2.log", filemode="a", level=logging.WARNING, format="%(message)s")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#* LR Feature extraction block (LRFB)
#> Input Image -> Conv(3, 4m) -> Conv(1, m) -> To FB
class LRFeatureExtractionBlock(nn.Module):
    def __init__(self, m, in_channels):
        super().__init__()
        self.m = m
        self.in_channels = in_channels
        self.conv_3x3 = nn.Conv2d(in_channels=self.in_channels, out_channels=4*self.m, kernel_size=3,padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=4*self.m)
        self.prelu_layer1 = nn.PReLU(4*self.m)

        self.conv_1x1 = nn.Conv2d(in_channels=4*self.m, out_channels=self.m, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=self.m)
        self.prelu_layer2 = nn.PReLU(self.m)


    def forward(self, image):
        image = self.conv_3x3(image)
        image = self.batch_norm1(image)
        image = self.prelu_layer1(image)

        image = self.conv_1x1(image)
        image = self.batch_norm2(image)
        image = self.prelu_layer2(image)
        return image


#* Reconstruction Block (RB)
#> FB Block -> Deconv(k, m) -> Conv(3, channel-out) -> to output block
class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels, k, m, stride, padding, out_channels=3):
        super().__init__()
        self.in_channel = in_channels
        self.k = k
        self.m = m
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        self.deconv = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.m, kernel_size=self.k, stride=self.stride, padding=self.padding)
        self.conv = nn.Conv2d(in_channels=self.m, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.prelu_layer = nn.PReLU(self.m)

    def forward(self, image):
        image = self.deconv(image)
        image = self.prelu_layer(image)
        image = self.conv(image)
        return image


#* Feedback Block (FB)
#! DO REFER TO IMAGE FOR MORE DETAILS
#> Concat( LRFB + FB(t-1) ) -> [Conv(1, m) -> Deconv(k, m) -> Conv(k, m) -> (Conv(1, m) -> Deconv(k, m) -> Conv(1, m) -> Conv(k, m) -> )*(G-1) ] -> To RB and to FB(t+1)
class FeebBackSubnetwork(nn.Module):
    def __init__(self, k, m, stride, padding, in_channels):
        super().__init__()
        self.k = k
        self.m = m
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels

        self.conv_1xm = nn.Conv2d(in_channels = self.in_channels, out_channels=self.m, kernel_size=1)
        self.deconv = nn.ConvTranspose2d(in_channels=self.m, out_channels=self.m, kernel_size=self.k, stride=self.stride, padding=self.padding)
        self.conv_kxm = nn.Conv2d(in_channels = self.m, out_channels = self.m, kernel_size=self.k, stride=self.stride, padding=self.padding)
        
        self.batch_norm = nn.BatchNorm2d(num_features=self.m)
        self.prelu_layer = nn.PReLU(self.m)


    def forward(self, prev_concat_L, prev_concat_H):
        # device = prev_concat_L.device
        
        # projection group going forward - Conv(1, m) -> Deconv(k, m) -> Conv(1, m) -> Conv(k, m)
        curr_input = prev_concat_L.detach().clone().to(device)

        curr_input = self.conv_1xm(curr_input)
        curr_input = self.batch_norm(curr_input)
        curr_input = self.prelu_layer(curr_input)

        curr_input = self.deconv(curr_input)
        curr_input = self.prelu_layer(curr_input)

        prev_concat_H = torch.cat([prev_concat_H, curr_input], dim=1)

        curr_input = prev_concat_H.detach().clone().to(device)

        curr_input = self.conv_1xm(curr_input)
        curr_input = self.prelu_layer(curr_input)
        curr_input = self.conv_kxm(curr_input)
        curr_input = self.prelu_layer(curr_input)

        prev_concat_L = torch.cat([prev_concat_L, curr_input], dim=1)
        
        return prev_concat_L, prev_concat_H



class FeedBackBlock(nn.Module):
    def __init__(self, k, m, stride, padding):
        super().__init__()
        self.k = k
        self.m = m
        self.stride = stride
        self.padding = padding
        
        self.prev_LR = None
        self.prev_HR = None
        
        
        self.conv_1xm = nn.Conv2d(in_channels = 2*self.m, out_channels=self.m, kernel_size=1)
        self.conv_1xm_last = nn.Conv2d(in_channels=7*self.m, out_channels=self.m, kernel_size=1)

        self.deconv = nn.ConvTranspose2d(in_channels=self.m, out_channels=self.m, kernel_size=self.k, stride=self.stride, padding=self.padding)
        self.conv_kxm = nn.Conv2d(in_channels = self.m, out_channels = self.m, kernel_size=self.k, stride=self.stride, padding=self.padding)
        
        self.batch_norm = nn.BatchNorm2d(num_features=self.m)
        self.prelu_layer = nn.PReLU(self.m)

        # tried using for loop, however shared the weights across the forward pass technically making it just feed-forward network G=1
        # Number of Projection Groups G=6 
        # self.group_block1 = FeebBackSubnetwork(self.k, self.m, self.stride, self.padding, self.m)
        self.group_block2 = FeebBackSubnetwork(self.k, self.m, self.stride, self.padding, 2*self.m)
        self.group_block3 = FeebBackSubnetwork(self.k, self.m, self.stride, self.padding, 3*self.m)
        self.group_block4 = FeebBackSubnetwork(self.k, self.m, self.stride, self.padding, 4*self.m)
        self.group_block5 = FeebBackSubnetwork(self.k, self.m, self.stride, self.padding, 5*self.m)
        self.group_block6 = FeebBackSubnetwork(self.k, self.m, self.stride, self.padding, 6*self.m)

    def forward(self, lr_image, prev_time_step_fb=None):
        # device = lr_image.device

        if prev_time_step_fb is None:
            #! take care for first time step input
            logging.info(f"LR-Image Size=Prev Time step = {lr_image.size()}")
            prev_time_step_fb = torch.zeros(lr_image.size()).to(device)
        
        concat_inputs = torch.cat([lr_image, prev_time_step_fb], dim=1)   # (1, 64, w, h)
        
        # first projection group - Conv(1, m) -> Deconv(k, m) -> Conv(k, m)
        concat_inputs = self.conv_1xm(concat_inputs)
        concat_inputs = self.batch_norm(concat_inputs)
        concat_inputs = self.prelu_layer(concat_inputs)

        self.prev_LR = concat_inputs.detach().clone().to(device)
        
        concat_inputs = self.deconv(concat_inputs)
        concat_inputs = self.prelu_layer(concat_inputs)

        self.prev_HR = concat_inputs.detach().clone().to(device)
        
        concat_inputs = self.conv_kxm(concat_inputs)
        self.prev_LR = torch.cat([self.prev_LR, concat_inputs], dim=1)
        
        del concat_inputs

        logging.info(f"Size of prev_LR={self.prev_LR.size()}")
        logging.info(f"Size of prev_HR={self.prev_HR.size()}")
        
        self.prev_LR, self.prev_HR = self.group_block2(self.prev_LR, self.prev_HR)
        self.prev_LR, self.prev_HR = self.group_block3(self.prev_LR, self.prev_HR)
        self.prev_LR, self.prev_HR = self.group_block4(self.prev_LR, self.prev_HR)
        self.prev_LR, self.prev_HR = self.group_block5(self.prev_LR, self.prev_HR)
        self.prev_LR, self.prev_HR = self.group_block6(self.prev_LR, self.prev_HR)
        
        del self.prev_HR
        self.prev_LR = self.conv_1xm_last(self.prev_LR)
        return self.prev_LR
        

class InceptionSubBlock(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

        self.conv_1x1 = nn.Conv2d(in_channels=2*self.m, out_channels=self.m, kernel_size=1)
        self.conv_1x3 = nn.Conv2d(in_channels=2*self.m, out_channels=self.m, kernel_size=1)
        self.conv_1x5 = nn.Conv2d(in_channels=2*self.m, out_channels=self.m, kernel_size=1)
        
        self.conv_3x3 = nn.Conv2d(in_channels=self.m, out_channels=2*self.m, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels=self.m, out_channels=2*self.m, kernel_size=5, padding=2)

        self.conv_merge = nn.Conv2d(in_channels=5*self.m, out_channels=2*self.m, kernel_size=1)

        self.prelu_layer = nn.PReLU(5*self.m)
        self.prelu_layer2 = nn.PReLU(2*self.m)


    def forward(self, img):
        img1 = self.conv_1x1(img)

        img2 = self.conv_1x3(img)
        img2 = self.conv_3x3(img2)
       
        img3 = self.conv_1x5(img)
        img3 = self.conv_5x5(img3)
        
        img = torch.cat([img1, img2, img3], dim=1)
        img = self.prelu_layer(img)
        
        img = self.conv_merge(img)
        img = self.prelu_layer2(img)
        return img


class InceptionFeatureExtractionBlock(nn.Module):
    def __init__(self, in_channels, m):
        super().__init__()
        self.in_channels = in_channels
        self.m = m

        self.init_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=2*self.m, kernel_size=3, padding=1)
        self.init_prelu = nn.PReLU(2*self.m)

        self.last_conv = nn.Conv2d(in_channels=2*self.m, out_channels=self.m, kernel_size=1)
        self.last_prelu = nn.PReLU(self.m)

        self.sub_block1 = InceptionSubBlock(self.m)
        self.sub_block2 = InceptionSubBlock(self.m)
    
    def forward(self, img):
        img = self.init_conv(img)
        img = self.init_prelu(img)

        img = self.sub_block1(img)
        img = self.sub_block2(img)
                
        img = self.last_conv(img)
        img = self.last_prelu(img)
        return img


#* Upsample kernel (Choice is arbritary)
#> LR image -> Bilinear kernel -> To output block


#* Output
#> Output = RB Block + Upsample kernel (T outputs)
class SRFBN(nn.Module):
    def __init__(self, scale_factor,m=32, T=4, in_channels = 3, out_channels = 3, fe_block="normal"):
        super().__init__()
        self.scale_factor = scale_factor
        self.stride, self.padding, self.k = get_hyper_parameters(self.scale_factor)
        self.m = m
        self.T = T
        self.in_channels = in_channels
        self.out_channels = out_channels

        
        if fe_block=="normal":
            self.lrfb = LRFeatureExtractionBlock(self.m, self.in_channels)
        elif fe_block =="inception":
            self.lrfb = InceptionFeatureExtractionBlock(self.in_channels, self.m)
        self.fb = FeedBackBlock(self.k, self.m, self.stride, self.padding)
        self.rb = ReconstructionBlock(self.m, self.k, self.m, self.stride, self.padding, self.out_channels)
        

    def forward(self, image):
        # device = image.device

        self.prev_time_step_fb = None
        upsampled_image = nn.functional.interpolate(image, scale_factor = self.scale_factor, mode="bilinear")
        output_images = []

        for time_step in range(self.T):
            logging.info(f"Starting Time step - {time_step+1}")
            logging.info(f"Starting LRFB-{time_step+1}")
            output_tmp = self.lrfb(image)  # (1, 32, 406, 508)
            logging.info(f"Completed LRFB-{time_step+1}")

            if time_step==0:
                logging.info(f"Size after LRFB = {output_tmp.size()}")
            
            logging.info(f"Starting FB-{time_step+1}")
            output_tmp = self.fb(output_tmp, self.prev_time_step_fb)  # (1, 32, 406, 508)
            logging.info(f"Completed FB-{time_step+1}")
            if time_step==0:
                logging.info(f"Size after FB = {output_tmp.size()}")

            self.prev_time_step_fb = output_tmp.detach().clone().to(device)

            
            logging.info(f"Starting RB-{time_step+1}")
            output_tmp = self.rb(output_tmp)
            logging.info(f"Completed RB-{time_step+1}")

            if time_step==0:
                logging.info(f"Output tmp- {output_tmp.size()}, Upsampled_Image-{upsampled_image.size()}, Original Image-{image.size()}")
            output_image = torch.add(output_tmp, upsampled_image)
            output_images.append(output_image)
        
        del self.prev_time_step_fb
        return output_images


def get_hyper_parameters(scale_factor):
    if scale_factor == 2:
        stride = 2
        padding = 2
        k = 6
    elif scale_factor == 3:
        stride = 3
        padding = 2
        k = 7
    elif scale_factor == 4:
        stride = 4
        padding = 2
        k = 8
    else:
        raise ValueError("Upscale Factor must be between 2 and 4")
    
    return stride, padding, k
