#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-09 22:36:51
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-08 18:16:07
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import torch

from typing import Type


DATASET_REGISTRY: dict[ str, Type[torch.utils.data.Dataset] ] = dict()


def register_dataset(name: str):
    assert name not in DATASET_REGISTRY
    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return wrapper

from .graph import *
from .node import *
from .edge import *
