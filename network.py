import torch
import cv2
import torch.nn as nn
import logging

#TODO SRFBN - 3DCV Project

#? HYPER-PARAMETERS
#? G=6, m=32, T=4 (SRFBN-L)
#! PReLu layer after all conv and deconv blocks except the last one


#* LR Feature extraction block (LRFB)
#> Input Image -> Conv(3, 4m) -> Conv(1, m) -> To FB

#* Feedback Block (FB)
#! DO REFER TO IMAGE FOR MORE DETAILS
#> Concat( LRFB + FB(t-1) ) -> [Conv(1, m) -> Deconv(k, m) -> Conv(k, m) -> (Conv(1, m) -> Deconv(k, m) -> Conv(1, m) -> Conv(k, m) -> )*(G-1) ] -> To RB and to FB(t+1)


#* Reconstruction Block (RB)
#> FB Block -> Deconv(k, m) -> Conv(3, channel-out) -> to output block


#* Upsample kernel (Choice is arbritary)
#> LR image -> Bilinear kernel -> To output block


#* Output
#> Output = RB Block + Upsample kernel (T outputs)

logging.basicConfig(filename="network.log", filemode="a", level=logging.WARNING)
#* LR Feature extraction block (LRFB)
class LRFeatureExtractionBlock(nn.Module):
    def __init__(self, m, in_channels):
        super().__init__()
        self.m = m
        self.in_channels = in_channels
        self.conv_3x3 = nn.Conv2d(in_channels=self.in_channels, out_channels=4*self.m, kernel_size=3,padding=1)
        self.conv_1x1 = nn.Conv2d(in_channels=4*self.m, out_channels=self.m, kernel_size=1)
        self.prelu_layer = nn.PReLU()


    def forward(self, image):
        image = self.conv_3x3(image)
        image = self.prelu_layer(image)
        image = self.conv_1x1(image)
        image = self.prelu_layer(image)
        return image


#* Reconstruction Block (RB)
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
        self.prelu_layer = nn.PReLU()

    def forward(self, image):
        image = self.deconv(image)
        image = self.prelu_layer(image)
        image = self.conv(image)
        return image


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
        self.prelu_layer = nn.PReLU()


    def forward(self, prev_concat_L, prev_concat_H):
        # projection group going forward - Conv(1, m) -> Deconv(k, m) -> Conv(1, m) -> Conv(k, m)
        curr_input = prev_concat_L.detach().clone().cuda()

        curr_input = self.prelu_layer(self.conv_1xm(curr_input))
        curr_input = self.prelu_layer(self.deconv(curr_input))
        
        prev_concat_H = torch.cat([prev_concat_H, curr_input], dim=1).cuda()

        curr_input = prev_concat_H.detach().clone().cuda()

        curr_input = self.conv_1xm(curr_input)
        curr_input = self.prelu_layer(curr_input)
        curr_input = self.conv_kxm(curr_input)
        curr_input = self.prelu_layer(curr_input)

        prev_concat_L = torch.cat([prev_concat_L, curr_input], dim=1).cuda()
        
        del curr_input
        return prev_concat_L.cuda(), prev_concat_H.cuda()



class FeedBackBlock(nn.Module):
    def __init__(self, k, m, G, stride, padding):
        super().__init__()
        self.k = k
        self.m = m
        self.G = G
        self.stride = stride
        self.padding = padding
        
        self.prev_LR = None
        self.prev_HR = None

        
        
        self.conv_1xm = nn.Conv2d(in_channels = 2*self.m, out_channels=self.m, kernel_size=1)
        self.conv_1xm_last = nn.Conv2d(in_channels=(self.G+1)*self.m, out_channels=self.m, kernel_size=1)

        self.deconv = nn.ConvTranspose2d(in_channels=self.m, out_channels=self.m, kernel_size=self.k, stride=self.stride, padding=self.padding)
        self.conv_kxm = nn.Conv2d(in_channels = self.m, out_channels = self.m, kernel_size=self.k, stride=self.stride, padding=self.padding)
        self.prelu_layer = nn.PReLU()

        self.group_block_list = []
        for group in range(1, self.G):
            group_block = FeebBackSubnetwork(self.k, self.m, self.stride, self.padding, (group+1)*self.m).cuda()
            self.group_block_list.append(group_block)

        # self.group_block = FeebBackSubnetwork(self.k, self.m, self.stride, self.padding, self.in_channels)


    def forward(self, lr_image, prev_time_step_fb=None):
        if prev_time_step_fb is None:
            #! take care for first time step input
            logging.info(f"LR-Image Size=Prev Time step = {lr_image.size()}")
            prev_time_step_fb = torch.zeros(lr_image.size()).cuda()
        
        concat_inputs = torch.cat([lr_image, prev_time_step_fb], dim=1)   # (1, 64, w, h)

        for group in range(self.G):
            logging.info(f"Starting Group -{group+1}")
            if group==0:
                # first projection group - Conv(1, m) -> Deconv(k, m) -> Conv(k, m)
                concat_inputs = self.conv_1xm(concat_inputs)
                concat_inputs = self.prelu_layer(concat_inputs)
                self.prev_LR = concat_inputs.detach().clone().cuda()
                concat_inputs = self.prelu_layer(self.deconv(concat_inputs))
                self.prev_HR = concat_inputs.detach().clone().cuda()
                concat_inputs = self.conv_kxm(concat_inputs)
                self.prev_LR = torch.cat([self.prev_LR, concat_inputs], dim=1).cuda()
                del concat_inputs
            else:
                logging.info(f"Size of prev_LR={self.prev_LR.size()}")
                logging.info(f"Size of prev_HR={self.prev_HR.size()}")
                group_block = self.group_block_list[group-1]
                self.prev_LR, self.prev_HR = group_block(self.prev_LR, self.prev_HR)
           
        
        del self.prev_HR
        self.prev_LR = self.conv_1xm_last(self.prev_LR)
        return self.prev_LR.cuda()
        

class SRFBN(nn.Module):
    def __init__(self, scale_factor, m=32, G=6, T=4, in_channels = 3, out_channels = 3):
        super().__init__()
        self.scale_factor = scale_factor
        self.stride, self.padding, self.k = get_hyper_parameters(self.scale_factor)
        self.m = m
        self.G = G
        self.T = T
        self.in_channels = in_channels
        self.out_channels = out_channels
        

        self.lrfb = LRFeatureExtractionBlock(self.m, self.in_channels)
        self.fb = FeedBackBlock(self.k, self.m, self.G, self.stride, self.padding)
        self.rb = ReconstructionBlock(self.m, self.k, self.m, self.stride, self.padding, self.out_channels)


    def forward(self, image, train=True):
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

            self.prev_time_step_fb = output_tmp.detach().clone().cuda()

            
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
