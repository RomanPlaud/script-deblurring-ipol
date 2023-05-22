#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Useful functions for initializations

Copyright (C) 2022, antoine.salmona@parisdescartes.fr, lucia.bouza-heguerte@u-paris.fr

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>
"""

import torch
import torch.nn as nn

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """ICNR initializer for checkerboard artifact free sub pixel convolution
    following https://arxiv.org/abs/1806.02658.
        
    Args:
        x (Tensor): Tensor of weights to be initialized.
        scale (int): scaling factor (for instance scale=2 doubles the size of the image).
        init_func (nn.init): init function.
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)


def init_network_(network,init_func=nn.init.kaiming_normal_):
    """Initialize the network with the desired initialization policy.
    
    Args:
        network (nn.Module): network to be initialized.
        init_func (nn.init): init function.
    """
    layers_list = list(network.children()) 
    for layer in layers_list:
        if len(list(layer.children()))>0:
            init_network_(layer,init_func) #we want a single layer
        else:
            conv = isinstance(layer,nn.Conv2d) #init conv only
            if conv:
                parameters = list(layer.parameters())
                trainable = parameters[0].requires_grad #check if the layer is trainable
                if trainable:
                    init_func(layer.weight)
                    if hasattr(layer,'bias') and hasattr(layer.bias,'data'):
                        layer.bias.data.fill_(0.) #if conv has bias init bias to zero

