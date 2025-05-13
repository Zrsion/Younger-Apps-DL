#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-01 11:14:21
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################
from torch import nn
from torch.nn import Embedding
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from younger_apps_dl.models import register_model


@register_model('sage')
class SAGE_NP(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, dropout_rate, total_layer_number=4):
        super(SAGE_NP, self).__init__()
        self.dropout_rate = dropout_rate

        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        
        self.layers = nn.ModuleList()

        dims = [node_dim]
        layer_number = total_layer_number - 1 # - 1 for the first layer

        middle_dim = 2 * hidden_dim 
        step_up = (middle_dim - node_dim) // (layer_number // 2)
        for i in range(layer_number // 2):
            next_dim = dims[-1] + step_up
            dims.append(next_dim)  

        step_down = (middle_dim - hidden_dim) // (layer_number - layer_number // 2)
        for i in range(layer_number // 2, layer_number):
            next_dim = dims[-1] - step_down
            dims.append(next_dim)
        dims.append(node_dict_size)
        print('debugging --- dims: ', dims)

        for i in range(total_layer_number):
            self.layers.append(SAGEConv(dims[i], dims[i+1]))

        print(self.layers)
        self.initialize_parameters()

    def forward(self, x, edge_index):
        x = self.node_embedding_layer(x).squeeze(1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        for index, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if index < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training) 
        return x

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)