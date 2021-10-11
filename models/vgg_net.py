#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2021-10-11 11:47:19
LastEditTime: 2021-10-11 11:47:20
LastEditors: Kun
Description: 
FilePath: /SpeakerRecognition-ResNet-GhostVLAD/models/vgg_net.py
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d

from models.vgg_modules import IdentityBlock2D, ConvBlock2D, eps_const_value, momentum_const_value


####################################################################################################

class VGGSmallRes34(nn.Module):
    def __init__(self):
        super(VGGSmallRes34, self).__init__()
        # ===============================================
        #            Convolution Block 1
        # ===============================================
        self.conv1_1_3x3_s1 = Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                                     stride=(1, 1), groups=1, bias=False)
        # nn.init.orthogonal_(self.conv1_1_3x3_s1.weight)
        self.conv1_1_3x3_s1_bn = BatchNorm2d(
            num_features=64, eps=eps_const_value, momentum=momentum_const_value)

        # ===============================================
        #            Convolution Section 2
        # ===============================================
        self.conv_block_2_a = ConvBlock2D(
            filters_list=[48, 48, 96], input=64, stride=(1, 1))
        self.identity_block_2_b = IdentityBlock2D(
            filters_list=[48, 48, 96], input=96)

        # ===============================================
        #            Convolution Section 3
        # ===============================================
        self.conv_block_3_a = ConvBlock2D(filters_list=[96, 96, 128], input=96)
        self.identity_block_3_b = IdentityBlock2D(
            filters_list=[96, 96, 128], input=128)
        self.identity_block_3_c = IdentityBlock2D(
            filters_list=[96, 96, 128], input=128)

        # ===============================================
        #            Convolution Section 4
        # ===============================================
        self.conv_block_4_a = ConvBlock2D(
            filters_list=[128, 128, 256], input=128)
        self.identity_block_4_b = IdentityBlock2D(
            filters_list=[128, 128, 256], input=256)
        self.identity_block_4_c = IdentityBlock2D(
            filters_list=[128, 128, 256], input=256)

        # ===============================================
        #            Convolution Section 5
        # ===============================================
        self.conv_block_5_a = ConvBlock2D(
            filters_list=[256, 256, 512], input=256)
        self.identity_block_5_b = IdentityBlock2D(
            filters_list=[256, 256, 512], input=512)
        self.identity_block_5_c = IdentityBlock2D(
            filters_list=[256, 256, 512], input=512)

    def forward(self, x):
        x = self.conv1_1_3x3_s1(F.pad(x, (3, 3, 3, 3)))
        x = self.conv1_1_3x3_s1_bn(x)
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=(
            2, 2), stride=(2, 2), padding=0, ceil_mode=False)

        x = self.conv_block_2_a(x)
        x = self.identity_block_2_b(x)

        x = self.conv_block_3_a(x)
        x = self.identity_block_3_b(x)
        x = self.identity_block_3_c(x)

        x = self.conv_block_4_a(x)
        x = self.identity_block_4_b(x)
        x = self.identity_block_4_c(x)

        x = self.conv_block_5_a(x)
        x = self.identity_block_5_b(x)
        x = self.identity_block_5_c(x)

        x = F.max_pool2d(x, kernel_size=(3, 1), stride=(
            2, 1), padding=0, ceil_mode=False)
        return x


####################################################################################################

class VGGLargeRes34(torch.nn.Module):
    def __init__(self):
        super(VGGLargeRes34, self).__init__()

        # ===============================================
        #            Convolution Block 1
        # ===============================================
        self.conv1_1_3x3_s1 = Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                                     stride=(2, 2), groups=1, bias=False)
        # nn.init.orthogonal_(self.conv1_1_3x3_s1.weight)
        self.conv1_1_3x3_s1_bn = BatchNorm2d(
            num_features=64, eps=eps_const_value, momentum=momentum_const_value)

        # ===============================================
        #            Convolution Section 2
        # ===============================================
        self.conv_block_2_a = ConvBlock2D(
            filters_list=[64, 64, 256], input=64, stride=(1, 1))
        self.identity_block_2_b = IdentityBlock2D(
            filters_list=[64, 64, 256], input=256)
        self.identity_block_2_c = IdentityBlock2D(
            filters_list=[64, 64, 256], input=256)

        # ===============================================
        #            Convolution Section 3
        # ===============================================
        self.conv_block_3_a = ConvBlock2D(
            filters_list=[128, 128, 512], input=256)
        self.identity_block_3_b = IdentityBlock2D(
            filters_list=[128, 128, 512], input=512)
        self.identity_block_3_c = IdentityBlock2D(
            filters_list=[128, 128, 512], input=512)

        # ===============================================
        #            Convolution Section 4
        # ===============================================
        self.conv_block_4_a = ConvBlock2D(
            filters_list=[256, 256, 1024], input=512, stride=(1, 1))
        self.identity_block_4_b = IdentityBlock2D(
            filters_list=[256, 256, 1024], input=1024)
        self.identity_block_4_c = IdentityBlock2D(
            filters_list=[256, 256, 1024], input=1024)

        # ===============================================
        #            Convolution Section 5
        # ===============================================
        self.conv_block_5_a = ConvBlock2D(
            filters_list=[512, 512, 2048], input=1024)
        self.identity_block_5_b = IdentityBlock2D(
            filters_list=[512, 512, 2048], input=2048)
        self.identity_block_5_c = IdentityBlock2D(
            filters_list=[512, 512, 2048], input=2048)

        # self.fc = nn.Conv2d(2048, 512, kernel_size=(1, 2), stride=(1, 1))
        # self.fc = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 2), stride=(1, 1),
        #                     groups=1, bias=True)

    def forward(self, x):
        x = self.conv1_1_3x3_s1(F.pad(x, (3, 3, 3, 3)))
        x = self.conv1_1_3x3_s1_bn(x)
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=(
            2, 2), stride=(2, 2), padding=0, ceil_mode=False)

        x = self.conv_block_2_a(x)
        x = self.identity_block_2_b(x)
        x = self.identity_block_2_c(x)

        x = self.conv_block_3_a(x)
        x = self.identity_block_3_b(x)
        x = self.identity_block_3_c(x)

        x = self.conv_block_4_a(x)
        x = self.identity_block_4_b(x)
        x = self.identity_block_4_c(x)

        x = self.conv_block_5_a(x)
        x = self.identity_block_5_b(x)
        x = self.identity_block_5_c(x)

        x = F.max_pool2d(x, kernel_size=(3, 1), stride=(
            2, 1), padding=0, ceil_mode=False)

        # r = self.fc(x)
        # return x, r
        return x


####################################################################################################

def vgg_res34_net(net_size="small"):
    if "small" == net_size:
        return VGGSmallRes34()
    elif "large" == net_size:
        return VGGLargeRes34()
    else:
        raise Exception("invalid VGG-Res34-Net size: {}".format(net_size))


if __name__ == '__main__':
    vgg_res34_small = VGGSmallRes34()
    print(vgg_res34_small)

    x = torch.rand(10, 1, 64, 26)
    print("x: ", x.shape)
    y = vgg_res34_small(x)
    print("res: ", y.shape)

    vgg_res34_large = VGGLargeRes34()
    print(vgg_res34_large)

    x = torch.rand(10, 1, 64, 26)
    print("x: ", x.shape)
    # y, r = vgg_res34_large(x)
    y = vgg_res34_large(x)
    # print("r: ", r.shape)
    print("res: ", y.shape)
