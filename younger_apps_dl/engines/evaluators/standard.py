#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-16 22:58:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-08 21:29:13
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

from younger_apps_dl.commons.utils import get_device_descriptor, get_model_parameters_number
from younger_apps_dl.commons.logging import equip_logger, logger
from younger_apps_dl.commons.checkpoint import load_checkpoint

from younger_apps_dl.engines import BaseEngine, register_engine


class StandardEvaluatorOptions(BaseModel):
    # Checkpoint Options
    checkpoint_filepath: pathlib.Path  = Field(..., description='Path to load checkpoint.')
    node_number: int  = Field(2, gt=1, description='Number of devices participating in distributed training. It should be > 1.')
    # Iteration Options
    batch_size: int = Field(32, ge=1, description='Batch size for validation.')


@register_engine('evaluator', 'standard')
class StandardEvaluator(BaseEngine[StandardEvaluatorOptions]):
    OPTIONS = StandardEvaluatorOptions

    def log(self, metric_names: list[str], metric_values: list[torch.Tensor], metric_formats: list[Callable[[float], str]]) -> None:
        logs = list()
        for metric_name, metric_value, metric_format in zip(metric_names, metric_values, metric_formats):
            logs.append(f'[{metric_name}]={metric_format(float(metric_value / self.options.node_number))}')
        logger.info(f'Evaluation Results - {" ".join(logs)}')

    def run(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        evaluate_fn: Callable[[torch.nn.Module, Any], tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]],
        dataloader_type: Literal['pth', 'pyg'] = 'pth',
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
        parameters_number = get_model_parameters_number(model)
        parameters_number_str = str()
        for name, number in parameters_number.items():
            parameters_number_str += f'{name}: {number} Elements ;\n'
        parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
        logger.info(
            f'\n======= v Model Architecture v ======='
            f'\n{model}'
            f'\n'
            f'\n====== v Number of Parameters v ======'
            f'\n{parameters_number_str}'
        )

        self.evaluate(
            model,
            dataset,
            evaluate_fn,
            dataloader_type
        )

    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        evaluate_fn: Callable[[torch.nn.Module, Any], tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]],
        dataloader_type: Literal['pth', 'pyg'] = 'pth',
    ) -> None:
        if dataloader_type == 'pth':
            from torch.utils.data import DataLoader
        if dataloader_type == 'pyg':
            from torch_geometric.loader import DataLoader

        dataloader = DataLoader(dataset, batch_size=self.options.batch_size, shuffle=False)

        logger.info(f'-> Dataset Size: {len(dataset)} | Batch Size: {self.options.batch_size} | Iteration Size: {len(dataloader)}')

        logger.info(f'-> Evaluating ...')
        tic = time.time()
        model.eval()
        with torch.no_grad():
            metric_names, metric_values, metric_formats = zip(*evaluate_fn(model, dataloader))
            self.log(metric_names, metric_values, metric_formats)

        toc = time.time()

        logger.info(f'-> Finished. Overall Time Cost = {toc-tic:.2f}s)')

