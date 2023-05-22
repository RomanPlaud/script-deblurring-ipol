#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Module implementing particular layers needed by the network

Copyright (C) 2022, antoine.salmona@parisdescartes.fr, lucia.bouza-heguerte@u-paris.fr

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>
"""

import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
import torch.nn.functional as F
from utils_unet.utils import icnr


class SelfAttention(nn.Module):
    """Self attention layer  (Notation from https://arxiv.org/pdf/1805.08318.pdf)."""
   
    def __init__(self,n_channels,spectral=False):
        """
        Args:
            n_channels (int): number of channel of the input tensor.
            spectral (boolean): if True apply spectral normalization 
                defined in https://arxiv.org/abs/1802.05957.
        """
        super().__init__()
        self.f = nn.Conv2d(n_channels,n_channels//8,1,bias=False)
        self.g = nn.Conv2d(n_channels,n_channels//8,1,bias=False)
        self.h = nn.Conv2d(n_channels,n_channels,1,bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        if spectral:
            self.f = spectral_norm(self.f)
            self.g = spectral_norm(self.g)
            self.h = spectral_norm(self.h)


    def forward(self,x):
        """
        Args:
            x (Tensor): feature maps (size: (B,C,W,H)).
        Returns:
            gamma*o + x (Tensor): self attention value scaled (learned)
                + input feature maps (residual) (size: (B,C,W,H)).
        """
        batch_size, n_channels, width, height = x.size()
        f_x = self.f(x).view(batch_size,-1,width*height).permute(0,2,1).contiguous()
        g_x = self.g(x).view(batch_size,-1,width*height)
        attention_map = F.softmax(torch.bmm(f_x,g_x), dim=-1) # Matrix product
        h_x = self.h(x).view(batch_size,-1,width*height)
        o = torch.bmm(h_x,attention_map)
        o = o.view(batch_size,n_channels,width,height)
        return self.gamma*o + x


        
class ConvBlock(nn.Module):
    """Convolutional Block (Conv + BatchNorm + ReLU)."""

    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=None,dilation=1, 
                 groups=1,bias=False,padding_mode='zeros',use_activ=True,
                 spectral=False,last=False):
        """
        Args:
            Args of nn.Conv2d +
            use_activ (boolean): if True use ReLU activation.
            spectral (boolean):  if True apply spectral normalization 
                defined in https://arxiv.org/abs/1802.05957.
            last (boolean): if true set bias on conv (last upsample layer).
        """
        super().__init__()
        self.bias = bias
        if padding is None:
            padding = (kernel_size - 1)//2
        if last:
            self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,dilation=dilation,padding=padding,groups=groups,bias=True,padding_mode=padding_mode)
            self.bias = True
        else:
            self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,dilation=dilation,padding=padding,groups=groups,bias=bias,padding_mode=padding_mode)
        if spectral and not last:
            self.conv = spectral_norm(self.conv)
        self.use_activ = use_activ
        if not self.bias:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        if self.use_activ:
            self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        out = self.conv(x)
        if self.use_activ:
            out = self.relu(out)
        if not self.bias: # No need to add bias if batch normalization
            out = self.batch_norm(out)
        return out



class UpSample(nn.Module):
    """UpSample layer following https://arxiv.org/abs/1609.05158."""

    def __init__(self,in_channels,out_channels,scale=2,blur=True,spectral=False,last=False):
        """
        Args:
            in_channels (int): number of channels of the input tensor.
            out_channels (int): number of channels of the output tensor.
            scale (int): scaling factor (for instance scale=2 doubles the size of the image).
            blur (boolean): if True corrects checkerboard artifacts 
                following https://arxiv.org/abs/1806.02658.
            spectral (boolean): if True applies spectral normalization 
                defined in https://arxiv.org/abs/1802.05957.
            last (boolean): if true set bias on conv (last upsample layer).
        """
        super().__init__()
        self.conv = ConvBlock(in_channels,out_channels*scale**2,1,last=last,
                              spectral=spectral,use_activ=False) # Sub pixel convolution
        icnr(self.conv.conv.weight) # Apply ICNR init
        self.shuf = nn.PixelShuffle(scale) # Transform a tensor of size (b,c*r^2,a,b) in size (b,c,r*a,r*b) (see article)
        self.blur = blur
        if self.blur:
            self.pad = nn.ReplicationPad2d((1,0,1,0)) 
            self.pool = nn.AvgPool2d(2, stride=1) # Apply blur to correct checkerboard artifacts
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        out = self.shuf(self.relu(self.conv(x)))
        return self.pool(self.pad(out)) if self.blur else x



class SigmoidRange(nn.Module):
    """Sigmoid module with range (low,high)."""
    
    def __init__(self, low, high):
        """
        Args:
            low (float or torch.Tensor): lower bound of the sigmoid.
            high (float or torch.Tensor): higher bound of the sigmoid.
        """
        super().__init__()
        self.low = low  # Lower bound
        self.high = high  # Higher bound

    def forward(self, x):
        if isinstance((self.low,self.high),(float,float)):
            out = torch.sigmoid(x) * (self.high - self.low) + self.low
        else:
            x = x.permute(0,2,3,1)
            out = torch.sigmoid(x).mul(other=self.high.to(x.device) - self.low.to(x.device)).add(other=self.low.to(x.device))
            out = out.permute(0,3,1,2)
        return out



