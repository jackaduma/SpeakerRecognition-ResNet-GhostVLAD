#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2021-10-11 11:46:18
LastEditTime: 2021-10-11 11:46:18
LastEditors: Kun
Description: 
FilePath: /SpeakerRecognition-ResNet-GhostVLAD/models/vgg_modules.py
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d

weight_decay = 1e-4
eps_const_value = 1e-05
momentum_const_value = 0.1


class ConvBlock2D(nn.Module):
    def __init__(self, filters_list, input, stride=(2, 2), **kwargs):
        super(ConvBlock2D, self).__init__()
        filters1, filters2, filters3 = filters_list

        self.conv_1x1_reduce = Conv2d(in_channels=input,
                                      out_channels=filters1,
                                      kernel_size=(1, 1),
                                      stride=stride, groups=1, bias=False)
        # nn.init.orthogonal_(self.conv_1x1_reduce.weight)
        self.conv_1x1_reduce_bn = BatchNorm2d(num_features=filters1,
                                              eps=eps_const_value,
                                              momentum=momentum_const_value)

        self.conv_1x1_proj = Conv2d(in_channels=input,
                                    out_channels=filters3,
                                    kernel_size=(1, 1),
                                    stride=stride, groups=1, bias=False)
        # nn.init.orthogonal_(self.conv_1x1_proj.weight)
        self.conv_1x1_proj_bn = BatchNorm2d(num_features=filters3,
                                            eps=eps_const_value,
                                            momentum=momentum_const_value)

        self.conv_3x3 = Conv2d(in_channels=filters1,
                               out_channels=filters2,
                               kernel_size=(3, 3),
                               stride=(1, 1), groups=1, bias=False)
        # nn.init.orthogonal_(self.conv_3x3.weight)
        self.conv_3x3_bn = BatchNorm2d(num_features=filters2,
                                       eps=eps_const_value,
                                       momentum=momentum_const_value)

        self.conv_1x1_increase = Conv2d(in_channels=filters2,
                                        out_channels=filters3,
                                        kernel_size=(1, 1),
                                        stride=(1, 1), groups=1, bias=False)
        # nn.init.orthogonal_(self.conv_1x1_increase.weight)
        self.conv_1x1_increase_bn = BatchNorm2d(num_features=filters3,
                                                eps=eps_const_value,
                                                momentum=momentum_const_value)

    def forward(self, x):
        x1 = self.conv_1x1_reduce(x)

        x2 = self.conv_1x1_proj(x)
        x2 = self.conv_1x1_proj_bn(x2)

        x1 = self.conv_1x1_reduce_bn(x1)
        x1 = F.pad(F.relu(x1, inplace=True), (1, 1, 1, 1))
        x1 = self.conv_3x3(x1)
        x1 = self.conv_3x3_bn(x1)
        x1 = self.conv_1x1_increase(F.relu(x1, inplace=True))
        x1 = self.conv_1x1_increase_bn(x1)

        return F.relu(x1 + x2, inplace=True)


class IdentityBlock2D(nn.Module):
    def __init__(self, filters_list, input, **kwargs):
        super(IdentityBlock2D, self).__init__()

        filters1, filters2, filters3 = filters_list

        self.conv_1x1_reduce = Conv2d(in_channels=input,
                                      out_channels=filters1,
                                      kernel_size=(1, 1),
                                      stride=(1, 1), groups=1, bias=False)
        # nn.init.orthogonal_(self.conv_1x1_reduce.weight)
        self.conv_1x1_reduce_bn = BatchNorm2d(num_features=filters1,
                                              eps=eps_const_value,
                                              momentum=momentum_const_value)

        self.conv_3x3 = Conv2d(in_channels=filters1,
                               out_channels=filters2,
                               kernel_size=(3, 3),
                               stride=(1, 1), groups=1, bias=False)
        # nn.init.orthogonal_(self.conv_3x3.weight)
        self.conv_3x3_bn = BatchNorm2d(num_features=filters2,
                                       eps=eps_const_value,
                                       momentum=momentum_const_value)

        self.conv_1x1_increase = Conv2d(in_channels=filters2,
                                        out_channels=filters3,
                                        kernel_size=(1, 1),
                                        stride=(1, 1), groups=1, bias=False)
        # nn.init.orthogonal_(self.conv_1x1_increase.weight)
        self.conv_1x1_increase_bn = BatchNorm2d(num_features=filters3,
                                                eps=eps_const_value,
                                                momentum=momentum_const_value)

    def forward(self, x):
        x1 = self.conv_1x1_reduce(x)
        x1 = self.conv_1x1_reduce_bn(x1)

        x1 = F.pad(F.relu(x1, inplace=True), (1, 1, 1, 1))
        x1 = self.conv_3x3(x1)
        x1 = self.conv_3x3_bn(x1)
        x1 = self.conv_1x1_increase(F.relu(x1, inplace=True))
        x1 = self.conv_1x1_increase_bn(x1)

        return F.relu(x1 + x, inplace=True)
