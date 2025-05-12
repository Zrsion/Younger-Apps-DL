#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-16 22:58:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-24 09:43:42
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import time
import torch
import pathlib

from typing import Any, Literal, Callable
from pydantic import BaseModel, Field

from younger_apps_dl.commons.utils import get_device_descriptor
from younger_apps_dl.commons.logging import equip_logger, logger
from younger_apps_dl.commons.checkpoint import load_checkpoint

from younger_apps_dl.engines import BaseEngine, register_engine


class RawInputOptions(BaseModel):
    load_dirpath: pathlib.Path = Field(..., description="Directory path to inputs on disk.")
    save_dirpath: pathlib.Path = Field(..., description="Directory path to outputs on disk.")


class StandardPredictorOptions(BaseModel):
    # Checkpoint Options
    checkpoint_filepath: pathlib.Path  = Field(..., description='Path to load checkpoint.')
    source: Literal['raw', 'api', 'cli'] = Field('raw', description='Source of input data for prediction. '
                                                                    '\'raw\' indicates data is loaded from disk; '
                                                                    '\'api\' indicates data comes from a live API call; '
                                                                    '\'cli\' indicates data is passed from command-line interface.')

    raw: RawInputOptions | None


@register_engine('predictor', 'standard')
class StandardPredictor(BaseEngine[StandardPredictorOptions]):
    OPTIONS = StandardPredictorOptions

    def run(
        self,
        model: torch.nn.Module,
        predict_raw_fn: Callable[[torch.nn.Module, pathlib.Path, pathlib.Path], None] | None,
        logging_filepath: pathlib.Path | None = None,
    ) -> None:
        equip_logger(logging_filepath)
        checkpoint = load_checkpoint(self.options.checkpoint_filepath)

        device_descriptor = get_device_descriptor('GPU', 0)
        model.to(device=device_descriptor)

        logger.info(f'-> Checkpoint from [Epoch/Step/Itr]@[{checkpoint.epoch}/{checkpoint.step}/{checkpoint.itr}].')

        logger.info(f'    v Loading Parameters ...')
        model.load_state_dict(checkpoint.model_state_dict)
        logger.info(f'    ^ Loaded.')

        if self.options.source == 'raw':
            self.predict_raw(
                model,
                predict_raw_fn,
            )

    def predict_raw(
        self,
        model: torch.nn.Module,
        predict_raw_fn: Callable[[torch.nn.Module, pathlib.Path, pathlib.Path], None],
    ) -> None:
        logger.info(f'-> Load Raw From: {self.options.raw.load_dirpath}')

        logger.info(f'-> Predicting ...')
        tic = time.time()
        model.eval()
        with torch.no_grad():
            predict_raw_fn(model, self.options.raw.load_dirpath, self.options.raw.save_dirpath)

        toc = time.time()

        logger.info(f'-> Finished. Overall Time Cost = {toc-tic:.2f}s)')