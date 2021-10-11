#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2021-09-30 18:32:40
LastEditTime: 2021-09-30 18:32:40
LastEditors: Kun
Description: 
FilePath: /SpeakerRecognition-ResNet-GhostVLAD/models/vlad_pooling.py
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostVLAD(nn.Module):
    def __init__(self, vlad_cluster, ghost_cluster, in_dim):
        super(GhostVLAD, self).__init__()
        self.vlad_cluster = vlad_cluster
        self.ghost_cluster = ghost_cluster
        self.conv1 = nn.Conv2d(in_dim, vlad_cluster +
                               ghost_cluster, kernel_size=(1, 1))
        self.centers = nn.Parameter(torch.rand(
            vlad_cluster + ghost_cluster, in_dim))

    def forward(self, x):
        N, C = x.shape[:2]

        soft_assign = self.conv1(x).view(
            N, self.vlad_cluster + self.ghost_cluster, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x = x.view(N, C, -1)
        x = x.expand(self.vlad_cluster + self.ghost_cluster, -
                     1, -1, -1).permute(1, 0, 2, 3)
        c = self.centers.expand(
            x.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        feat_res = x - c
        weighted_res = feat_res * soft_assign.unsqueeze(2)
        cluster_res = weighted_res.sum(dim=-1)

        cluster_res = cluster_res[:, :self.vlad_cluster, :]  # ghost
        cluster_res = F.normalize(cluster_res, p=2, dim=-1)
        vlad_feats = cluster_res.view(N, -1)
        # vlad_feats = F.normalize(vlad_feats, p=2, dim=-1)
        # print(vlad_feats.shape)
        return vlad_feats
