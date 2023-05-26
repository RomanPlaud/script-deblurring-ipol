#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Module implementing the network

Copyright (C) 2022, antoine.salmona@parisdescartes.fr, lucia.bouza-heguerte@u-paris.fr

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from utils_unet.layers import ConvBlock, UpSample, SelfAttention, SigmoidRange


class UnetUpBlock(nn.Module):
    """Up Block of the U-net composed of an upsample block and a conv block."""

    def __init__(self,in_channels_direct,in_channels_skip,out_channels,blur=True,spectral=False,
                 self_attention=False):
        """
        Args:
            in_channels_direct (int): number of channels from previous layer.
            in_channels_skip (int): number of channels from skip connection.
            out_channels (int): number of channels from skip connection.
            blur (boolean): if True corrects checkerboard artifacts 
                following https://arxiv.org/abs/1806.02658.
            spectral (boolean): if True applies spectral normalization
                defined in https://arxiv.org/abs/1802.05957.
            self_attention (boolean): if True adds a SelfAttention layer
                (see https://arxiv.org/pdf/1805.08318.pdf).
        """
        super().__init__()
        self.upsample = UpSample(in_channels_direct,out_channels,blur=blur,spectral=spectral)
        self.batch_norm = nn.BatchNorm2d(in_channels_skip)
        in_channels = out_channels + in_channels_skip
        self.relu = nn.ReLU()
        self.conv = ConvBlock(in_channels,out_channels,3,spectral=spectral)
        self.sa = self_attention
        if self.sa:
            self.self_attention = SelfAttention(out_channels,spectral=spectral)


    def forward(self,x_direct,x_skip):
        """
        Args:
            x_direct (Tensor): output of the previous layer.
            x_skip (Tensor): feature maps from the skip connection.
        Returns:
            out (Tensor): output tensor.
        """
        up_out = self.upsample(x_direct)
        if x_skip.shape[-2:]!=up_out.shape[-2:]:
            up_out = F.interpolate(up_out, x_skip.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out,self.batch_norm(x_skip)],dim=1))
        out = self.conv(cat_x)
        if self.sa:
            out = self.self_attention(out)
        return out



class Bridge(nn.Module):
    """Conv Bridge between the downsampling part and the upsampling part."""

    def __init__(self,channels,spectral=False):
        """
        Args:
            channels (int): number of channels of the input tensor.
            spectral (boolean): if True apply spectral normalization
                defined in https://arxiv.org/abs/1802.05957.
        """
        super().__init__()
        self.conv1 = ConvBlock(channels,2*channels,3,spectral=spectral)
        self.conv2 = ConvBlock(2*channels,channels,3,spectral=spectral)

    
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out



class Unet(nn.Module):
    """Main module: the U-shaped network using a resnet backbone."""
    DEPTH = 6

    yuv_from_rgb = torch.Tensor([[0.299, 0.587, 0.114],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])

    def __init__(self,n_channels=3,y_range=(torch.Tensor([-3.,-3.]),torch.Tensor([3.,3.])),backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
                 spectral=False,blur=True,pretrained=None, freeze=True):
        """
        Args:
            n_channels (int): number of channels of the output tensor.
            y_range (tuple): bounds for SigmoidRange.
            backbone (torchvision.models): resnet version (18, 34, 50, 101,or 152) 
                to be used as backbone.
            spectral (boolean): if True applies spectral normalization
                defined in https://arxiv.org/abs/1802.05957.
            blur (boolean): if True corrects checkerboard artifacts 
                following https://arxiv.org/abs/1806.02658.
        """
        super().__init__()
        self.n_channels = n_channels
        down_blocks = list()
        up_blocks = list()
        self.input_block = nn.Sequential(*list(backbone.children()))[:3]
        self.input_pool = list(backbone.children())[3]
        for bottleneck in list(backbone.children()):
            if isinstance(bottleneck,nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(True)
        self.bridge = Bridge(2048,spectral=spectral)
        up_blocks.append(UnetUpBlock(2048,1024,512,spectral=spectral,blur=blur))
        up_blocks.append(UnetUpBlock(512,512,512,spectral=spectral,blur=blur,self_attention=True))
        up_blocks.append(UnetUpBlock(512,256,512,spectral=spectral,blur=blur))
        up_blocks.append(UnetUpBlock(512,64,256,spectral=spectral,blur=blur))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.last_upsample = UpSample(256,256,blur=blur,last=True,spectral=spectral)
        self.res_block = nn.Sequential(ConvBlock(259,259,3,spectral=spectral,bias=True),
                                       ConvBlock(259,259,3,spectral=spectral,bias=True))
        self.dense = ConvBlock(259,n_channels,1,use_activ=False,bias=True,spectral=spectral)
        self.sigmoid = SigmoidRange(*y_range)
        
        #the following freezes already pretrained layers 
        if freeze:
            for param in self.input_block.parameters():
                param.requires_grad = False
            for param in self.down_blocks.parameters():
                param.requires_grad = False
        
        self.pretrained = pretrained is not None
        if pretrained is not None:
            state_dict = torch.load(pretrained,map_location='cpu')['state_dict']
            self.load_state_dict(state_dict,strict=True)


    def forward(self,x):
        if self.n_channels==2: #if n_channels==2 predict chrominances of YUV space
            x_yuv = x.permute(0,2,3,1)
            x_yuv = torch.matmul(x_yuv,Unet.yuv_from_rgb.t().to(x.device)).permute(0,3,1,2)
            Y = x_yuv[:,:1,:,:] #retrieve luminance channel

        pre_pools = dict() #dict for skip_connections
        pre_pools[f"layer_0"] = x #store x for skip_connections
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x  #store x for skip_connections
        x = self.input_pool(x)
        for i, block in enumerate(self.down_blocks, 2): #downsampling part
            x = block(x)
            if i == (Unet.DEPTH - 1):
                continue #we can't go deeper
            pre_pools[f"layer_{i}"] = x #store x for skip_connections
        
        x = self.bridge(self.relu(self.bn(x)))

        for i, block in enumerate(self.up_blocks, 1): #upsampling part
            key = f"layer_{Unet.DEPTH - 1 - i}" 
            x = block(x,pre_pools[key]) #skip_connections
        x = self.last_upsample(x) 
        x = torch.cat([x,pre_pools[f"layer_0"]],1) #last skip connection
        x = self.res_block(x) + x #residual learning
        x = self.dense(x)
        x = self.sigmoid(x) 
        if self.n_channels==2: #concat with luminance channel and convert back in RGB
            x = torch.cat([Y,x],1)
            x = x.permute(0,2,3,1)
            x = torch.matmul(x,Unet.yuv_from_rgb.inverse().t().to(x.device)).permute(0,3,1,2)            
            
        del pre_pools
   
        return x

import cv2
import datetime
from utils_unet.utils_unet import downsampling, forward_model, upsampling
# from utils_unet.unet import Unet

import os
from tqdm import trange, tqdm
from PIL import Image
from torch import multiprocessing
import torch 
from torchvision import models


vcapture = cv2.VideoCapture("example_videos/vid_example.mp4")
length = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
dim = (width, height)
fps = vcapture.get(cv2.CAP_PROP_FPS)

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = os.path.join("output_videos/", f"video_blurred_{current_time}.avi")

vwriter = cv2.VideoWriter(file_name,
                            cv2.VideoWriter_fourcc('F','M','P','4'),
                            fps, dim)



size_img = (192, 192)
def downsampling_bis(x):
    return downsampling(x, size_img=size_img)
n_jobs = 1

results = []

for _ in trange(0, length):
    imgs = []
    success, img = vcapture.read()
    if not success : 
        break
    imgs.append(Image.fromarray(img[:,:,[2,1,0]]))

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=n_jobs)
    result = pool.map(downsampling_bis, imgs)
    pool.close()
    pool.join()





# backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# y_range=torch.Tensor([-3.,-3.,-3.]), torch.Tensor([3.,3.,3.])
# path_model = "weights/deblurring_model.pth"
# model = Unet(n_channels=3, pretrained=path_model, backbone=backbone, y_range=y_range, spectral=True)

# outputs = []
# for result in results:

#     input_img  = torch.cat([x[1] for x in result], dim=0)

    
#     output = forward_model(input_img, model, device='cuda')

#     outputs.append(output)

# def upsampling_bis(x, y, z):
#     return upsampling(x, y, z, dim=dim)

# if __name__ == '__main__':
#     pool = multiprocessing.Pool(processes=n_jobs)
#     for result, output in tqdm(zip(results, outputs)):


#         inputs_arrays = [x[0] for x in result]
#         input_imgs = [x[1] for x in result]
#         out = [outp for outp in output]



#         aaa = pool.starmap(upsampling_bis, zip(inputs_arrays, input_imgs, out))

#     pool.close()
#     pool.join()