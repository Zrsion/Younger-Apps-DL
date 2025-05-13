#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-02-21 13:19:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-23 11:13:59
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib

from younger.commons.logging import set_logger, use_logger, get_logger

from younger_apps_dl import __thename__

def equip_logger(logging_filepath: pathlib.Path | str | None = None):
    set_logger(__thename__, mode='both', level='INFO', logging_filepath=logging_filepath)
    use_logger(__thename__)

logger = get_logger(__thename__)