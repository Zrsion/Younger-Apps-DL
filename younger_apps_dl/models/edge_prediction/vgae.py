#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-01 10:46:56
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch_geometric.nn import GCNConv

from younger_apps_dl.models import register_model

@register_model('vgae_ep')
class VGAE_EP(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, output_dim):
        super(VGAE_EP, self).__init__()
        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        self.conv_1 = GCNConv(node_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, output_dim)
        self.conv_logvar = GCNConv(hidden_dim, output_dim)
        self.initialize_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv_1(x, edge_index)
        x = F.relu(x)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)