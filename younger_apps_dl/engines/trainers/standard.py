#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-16 22:58:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-24 17:01:20
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import time
import torch
import pathlib

from torch import distributed
from typing import Any, Literal, Callable
from pydantic import BaseModel, Field

from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from younger.commons.io import create_dir
from younger.commons.utils import no_operation

from younger_apps_dl.commons.utils import get_device_descriptor, make_reproducible, broadcast_object
from younger_apps_dl.commons.logging import logger, equip_logger
from younger_apps_dl.commons.checkpoint import load_checkpoint, save_checkpoint, Checkpoint

from younger_apps_dl.engines import BaseEngine, register_engine


class StandardTrainerOptions(BaseModel):
    # Checkpoint Options
    checkpoint_savepath: pathlib.Path = Field(..., description='Directory path to save checkpoint.')
    checkpoint_basename: str = Field('checkpoint', description='Base name of the checkpoint for save/load.')
    checkpoint_keepdisk: int = Field(5, ge=1, description='Number of checkpoints to keep on disk.')

    ## Resume Options
    resume_filepath: pathlib.Path | None  = Field(None, description='Path to load checkpoint. If None, train from scratch.')
    reset_iteration: bool = Field(True, description='Whether to reset the iteration status (epoch, step) when loading a checkpoint.')
    reset_optimizer: bool = Field(True, description='Whether to reset the optimizer when loading a checkpoint.')
    reset_scheduler: bool = Field(True, description='Whether to reset the scheduler when loading a checkpoint.')

    # Iteration Options
    seed: int = Field(3407, ge=0, description='Random seed for reproducibility.')
    shuffle: bool = Field(True, description='Shuffle the training data each epoch.')
    life_cycle: int = Field(100, ge=1, description='Lefe cycle of the training process (in epochs).')

    report_period: int = Field(100, ge=1, description='Period (in steps) to report the training status.')
    update_period: int = Field(1, ge=1, description='Period (in steps) to update the model parameters.')
    saving_period: int = Field(1000, ge=1, description='Period (in steps) to save the model parameters.')

    train_batch_size: int = Field(32, ge=1, description='Batch size for training.')
    valid_batch_size: int = Field(32, ge=1, description='Batch size for validation.')

    early_stop_enable: bool = Field(False, description='Stop training early if the metric no longer improves.')
    early_stop_target: Literal['min', 'max'] = Field('min', description='Whether the monitored metric should be minimized or maximized.')
    early_stop_metric: str = Field('loss', description='The name of the metric to monitor for early stopping.')
    early_stop_patience: int = Field(10, ge=1, description='Number of evaluation round to wait for an improvement before stopping.')
    early_stop_tolerance: float = Field(0.05, ge=0, description='Minimum change in the monitored metric to qualify as an improvement.')

    # Distribution Options
    distributed: bool = Field(False, description='Whether to use distributed training. If False, the options about distributed training will take no effect.')
    master_addr: str  = Field('localhost', description='Master address for distributed training.')
    master_port: str  = Field('16161', description='Master port for distributed training.')
    master_rank: int  = Field(0, ge=0, description='Master rank for distributed training. It should be < world_size and >= 0.')
    node_number: int  = Field(2, gt=1, description='Number of devices participating in distributed training. It should be > 1.')

    worker_number: int = Field(10, description='Number of workers for dataloader.')


@register_engine('trainer', 'standard')
class StandardTrainer(BaseEngine[StandardTrainerOptions]):
    OPTIONS = StandardTrainerOptions

    def log(self, epoch: int, step: int, itr: int, metrics: list[tuple[str, torch.Tensor | float, Callable[[float], str]]], stage: Literal['train', 'valid']) -> None:
        with torch.no_grad():
            logs = list()
            for metric_name, metric_value, metric_format in metrics:
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.detach()
                if stage == 'train' and self.options.distributed:
                    distributed.all_reduce(metric_value, op = distributed.ReduceOp.SUM)
                    metric_value = metric_value / self.options.node_number
                logs.append(f'[{metric_name}]={metric_format(float(metric_value))}')
            logger.info(f'   [Epoch/Step/Itr]@[{epoch}/{step}/{itr}] - {" ".join(logs)}')

    def run(
        self,
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        train_fn: Callable[[torch.nn.Module, Any], list[tuple[str, torch.Tensor | float, Callable[[float], str]]]],
        valid_fn: Callable[[torch.nn.Module, Any], list[tuple[str, torch.Tensor | float, Callable[[float], str]]]],
        on_step_begin_fn: Callable[[int], None] = no_operation,
        on_step_end_fn: Callable[[int], None] = no_operation,
        on_epoch_begin_fn: Callable[[int], None] = no_operation,
        on_epoch_end_fn: Callable[[int], None] = no_operation,
        dataloader_type: Literal['pth', 'pyg'] = 'pth',
        logging_filepath: pathlib.Path | None = None,
    ) -> None:
        """
        _summary_

        :param model: _description_
        :type model: torch.nn.Module
        :param optimizer: _description_
        :type optimizer: torch.optim.Optimizer
        :param scheduler: _description_
        :type scheduler: torch.optim.lr_scheduler.LRScheduler
        :param train_dataset: _description_
        :type train_dataset: torch.utils.data.Dataset
        :param valid_dataset: _description_
        :type valid_dataset: torch.utils.data.Dataset
        :param train_fn: A callable that defines how to process a `minibatch` during training, feeding it to the model and returning the computed metrics (names, values, formats). The 0th metric must be the loss to be optimized.
        :type train_fn: Callable[[torch.nn.Module, Any], list[tuple[str, torch.Tensor | float, Callable[[float], str]]]]
        :param valid_fn: A callable that defines how to process a `sequence` of `minibatch` during validation, feeding it to the model and returning the computed metrics (names, values, formats).
        :type valid_fn: Callable[[torch.nn.Module, Any], list[tuple[str, torch.Tensor | float, Callable[[float], str]]]]
        :param on_step_end_fn: _description_
        :type on_step_end_fn: Callable[[int], None]
        :param on_epoch_end_fn: _description_
        :type on_epoch_end_fn: Callable[[int], None]
        :param dataloader: _description_, defaults to 'pth'
        :type dataloader: Literal[&#39;pth&#39;, &#39;pyg&#39;], optional
        """

        start_from_epoch: int = 0
        start_from_step: int = 0
        start_from_itr: int = 0
        equip_logger(logging_filepath)
        print(logger)
        print(logger.handlers)
        if self.options.resume_filepath is None:
            logger.info(f'-> Train from scratch.')
        else:
            checkpoint = load_checkpoint(self.options.resume_filepath)

            logger.info(f'-> Train from [Epoch/Step/Itr]@[{checkpoint.epoch}/{checkpoint.step}/{checkpoint.itr}].')

            if self.options.reset_iteration:
                logger.info(f'   Reset Epoch & Step & Itr.')
            else:
                start_from_epoch: int = checkpoint.epoch
                start_from_step: int = checkpoint.step
                start_from_itr: int = checkpoint.itr

            if self.options.reset_optimizer:
                logger.info(f'   Reset Optimizer.')
            else:
                optimizer.load_state_dict(checkpoint.optimizer_state_dict)

            if self.options.reset_scheduler:
                logger.info(f'   Reset Scheduler.')
            else:
                scheduler.load_state_dict(checkpoint.scheduler_state_dict)

            logger.info(f'    v Loading Parameters ...')
            model.load_state_dict(checkpoint.model_state_dict)
            logger.info(f'    ^ Loaded.')

        create_dir(self.options.checkpoint_savepath)
        logger.info(f'-> Checkpoints will be saved into: \'{self.options.checkpoint_savepath}\'')

        logger.info(f'-> Training Start ...')
        logger.info(f'   Train Life Cycle: Total {self.options.life_cycle} Epochs!')
        logger.info(f'   Report every {self.options.report_period} Iteration (Itr);')
        logger.info(f'   Update every {self.options.update_period} Iteration (Itr);')
        logger.info(f'   Saving every {self.options.saving_period} Iteration (Itr);')

        tic = time.time()
        if self.options.distributed:
            master_rank = self.options.master_rank
            master_addr = self.options.master_addr
            master_port = self.options.master_port
            node_number = self.options.node_number

            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port

            assert torch.cuda.device_count() >= node_number, f'Insufficient GPU: {torch.cuda.device_count()}'
            assert master_rank < node_number, f'Wrong Master Rank: {master_rank}'

            logger.info(f'-> Distributed training - Total {node_number} GPU used.')

            torch.multiprocessing.spawn(self.pile_train, args=(
                start_from_epoch, start_from_step, start_from_itr,
                model, optimizer, scheduler,
                train_dataset, valid_dataset,
                train_fn, valid_fn,
                on_step_begin_fn,
                on_step_end_fn,
                on_epoch_begin_fn,
                on_epoch_end_fn,
                dataloader_type,
                logging_filepath,
            ), nprocs=node_number, join=True)
        else:
            self.solo_train(
                start_from_epoch, start_from_step, start_from_itr,
                model, optimizer, scheduler,
                train_dataset, valid_dataset,
                train_fn, valid_fn,
                on_step_begin_fn,
                on_step_end_fn,
                on_epoch_begin_fn,
                on_epoch_end_fn,
                dataloader_type,
                logging_filepath,
            )
        toc = time.time()
        logger.info(f'-> All Done! Overall Time Cost = {toc-tic:.2f}s')

    def pile_train(
        self, rank: int,
        start_from_epoch: int, start_from_step: int, start_from_itr: int,
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        train_fn: Callable[[torch.nn.Module, Any], list[tuple[str, torch.Tensor | float, Callable[[float], str]]]],
        valid_fn: Callable[[torch.nn.Module, Any], list[tuple[str, torch.Tensor | float, Callable[[float], str]]]],
        on_step_begin_fn: Callable[[int], None] = no_operation,
        on_step_end_fn: Callable[[int], None] = no_operation,
        on_epoch_begin_fn: Callable[[int], None] = no_operation,
        on_epoch_end_fn: Callable[[int], None] = no_operation,
        dataloader_type: Literal['pth', 'pyg'] = 'pth',
        logging_filepath: pathlib.Path | None = None,
    ) -> None:
        equip_logger(logging_filepath)

        make_reproducible(self.options.seed)
        torch.autograd.set_detect_anomaly(True)

        device_descriptor = get_device_descriptor('GPU', rank)
        torch.cuda.set_device(rank)
        model.to(device=device_descriptor)
        logger.info(f'-> Process {rank} Use Device \'{device_descriptor}\'')

        os.getpid()
        master_flag = rank == self.options.master_rank
        if master_flag:
            logger.disabled = False
        else:
            logger.disabled = True

        distributed.init_process_group('nccl', rank=rank, world_size=self.options.node_number)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=self.options.node_number, seed=self.options.seed, shuffle=self.options.shuffle, drop_last=True)

        if dataloader_type == 'pth':
            from torch.utils.data import DataLoader
        if dataloader_type == 'pyg':
            from torch_geometric.loader import DataLoader

        train_dataloader = DataLoader(train_dataset, batch_size=self.options.train_batch_size, sampler=train_sampler, pin_memory=True, persistent_workers=True, num_workers=self.options.worker_number)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.options.valid_batch_size, shuffle=False)

        early_stop = False
        if self.options.early_stop_enable:
            early_stop_progress = 0
            early_stop_best_value = float('inf')
        model.train()
        optimizer.zero_grad()
        itr = start_from_itr
        # Epoch 1 .. self.options.life_cycle
        for epoch in range(1, self.options.life_cycle + 1):
            if epoch < start_from_epoch:
                logger.info(f'-> Skip Epoch {epoch} Before Epoch {start_from_epoch}.')
                continue

            train_sampler.set_epoch(epoch)

            tic = time.time()
            on_epoch_begin_fn(epoch)
            # Step 1 .. len(train_dataloader)
            for step, minibatch in enumerate(train_dataloader, start=1):
                on_step_begin_fn(step)
                if epoch == start_from_epoch and step <= start_from_step:
                    logger.info(f'-> Skip Mini-Batch {step} Before Step {start_from_step}.')
                    continue

                itr = itr + 1

                metrics = train_fn(model, minibatch)
                # Delegate backward to `train_fn` for cases like 1 forward + N backward.
                # metrics[0][1].backward()

                # Update Model Parameters
                if itr % self.options.update_period == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                # Report Metrics
                if itr % self.options.report_period == 0:
                    self.log(epoch, step, itr, metrics, 'train')

                # Validate and Save Model
                if itr % self.options.saving_period == 0:
                    if master_flag:
                        logger.info(f'-> Validating ...')
                        model.eval()
                        stic = time.time()
                        with torch.no_grad():
                            metrics = valid_fn(model, valid_dataloader)
                        stoc = time.time()
                        self.log(epoch, step, itr, metrics, 'train')
                        model.train()
                        logger.info(f'   Time Cost: {stoc-stic:.2f}s')

                        logger.info(f'-> Saving ...')
                        stic = time.time()
                        checkpoint = Checkpoint(epoch, step, itr, model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict(), dict(((metric[0], metric[1]) for metric in metrics)))
                        save_checkpoint(checkpoint, self.options.checkpoint_savepath, self.options.checkpoint_basename, self.options.checkpoint_keepdisk)
                        stoc = time.time()
                        logger.info(f'   Time Cost: {stoc-stic:.2f}s')

                        if self.options.early_stop_enable:
                            if self.options.early_stop_target == 'min':
                                value = checkpoint.metrics[self.options.early_stop_metric]
                            if self.options.early_stop_target == 'max':
                                value = checkpoint.metrics[self.options.early_stop_metric] * -1

                            if value < early_stop_best_value - self.options.early_stop_tolerance:
                                early_stop_progress = 0
                                early_stop_best_value = value
                            else:
                                early_stop_progress = early_stop_progress + 1

                            if early_stop_progress == self.options.early_stop_patience:
                                early_stop = broadcast_object(True, rank)

                on_step_end_fn(step)
                if early_stop:
                    break

            on_epoch_end_fn(epoch)
            if early_stop:
                break
            toc = time.time()
            logger.info(f'-> Epoch@{epoch} Finished. Time Cost = {toc-tic:.2f}s')

        distributed.destroy_process_group()

    def solo_train(
        self,
        start_from_epoch: int, start_from_step: int, start_from_itr: int,
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        train_fn: Callable[[torch.nn.Module, Any], list[tuple[str, torch.Tensor | float, Callable[[float], str]]]],
        valid_fn: Callable[[torch.nn.Module, Any], list[tuple[str, torch.Tensor | float, Callable[[float], str]]]],
        on_step_begin_fn: Callable[[int], None] = no_operation,
        on_step_end_fn: Callable[[int], None] = no_operation,
        on_epoch_begin_fn: Callable[[int], None] = no_operation,
        on_epoch_end_fn: Callable[[int], None] = no_operation,
        dataloader_type: Literal['pth', 'pyg'] = 'pth',
        logging_filepath: pathlib.Path | None = None,
    ) -> None:
        equip_logger(logging_filepath)

        make_reproducible(self.options.seed)
        torch.autograd.set_detect_anomaly(True)

        device_descriptor = get_device_descriptor('GPU', 0)
        model.to(device=device_descriptor)

        train_sampler = RandomSampler(train_dataset) if self.options.shuffle else None
        if dataloader_type == 'pth':
            from torch.utils.data import DataLoader
        if dataloader_type == 'pyg':
            from torch_geometric.loader import DataLoader

        train_dataloader = DataLoader(train_dataset, batch_size=self.options.train_batch_size, sampler=train_sampler, pin_memory=True, persistent_workers=True, num_workers=self.options.worker_number)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.options.valid_batch_size, shuffle=False)

        best_loss = float('inf')
        best_epoch = 0
        best_step = 0
        best_itr = 0

        early_stop = False
        if self.options.early_stop_enable:
            early_stop_progress = 0
            early_stop_best_value = float('inf')
        model.train()
        optimizer.zero_grad()
        itr = start_from_itr
        # Epoch 1 .. self.options.life_cycle
        for epoch in range(1, self.options.life_cycle + 1):
            if epoch < start_from_epoch:
                logger.info(f'-> Skip Epoch {epoch} Before Epoch {start_from_epoch}.')
                continue

            tic = time.time()
            on_epoch_begin_fn(epoch)
            # Step 1 .. len(train_dataloader)
            for step, minibatch in enumerate(train_dataloader, start=1):
                on_step_begin_fn(epoch)
                if epoch == start_from_epoch and step <= start_from_step:
                    logger.info(f'-> Skip Mini-Batch {step} Before Step {start_from_step}.')
                    continue

                itr = itr + 1

                metrics = train_fn(model, minibatch)
                # Delegate backward to `train_fn` for cases like 1 forward + N backward.
                # metrics[0][1].backward()

                # Update Model Parameters
                if itr % self.options.update_period == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                # Report Metrics
                if itr % self.options.report_period == 0:
                    self.log(epoch, step, itr, metrics, 'train')

                # Validate and Save Model
                if itr % self.options.saving_period == 0:
                    logger.info(f'-> Validating ...')
                    model.eval()
                    stic = time.time()
                    with torch.no_grad():
                        metrics = valid_fn(model, valid_dataloader)

                    stoc = time.time()
                    self.log(epoch, step, itr, metrics, 'train')
                    model.train()
                    logger.info(f'   Time Cost: {stoc-stic:.2f}s')

                    logger.info(f'-> Saving checkpoint ...')
                    stic = time.time()
                    checkpoint = Checkpoint(epoch, step, itr, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), dict(((metric[0], metric[1]) for metric in metrics)))
                    save_checkpoint(checkpoint, self.options.checkpoint_savepath, self.options.checkpoint_basename, self.options.checkpoint_keepdisk)
                    stoc = time.time()
                    
                    logger.info(f'   Time Cost: {stoc-stic:.2f}s')

                    val_loss =  metrics[0][1]
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_epoch = epoch
                        best_step = step
                        best_itr = itr

                    logger.info(f"Up to now, Best checkpoint info: [Epoch {best_epoch}] [Step {best_step}] [Itr {best_itr}] | best_loss={best_loss:.4f}")
                    
                    if self.options.early_stop_enable:
                        if self.options.early_stop_target == 'min':
                            value = checkpoint.metrics[self.options.early_stop_metric]
                        if self.options.early_stop_target == 'max':
                            value = checkpoint.metrics[self.options.early_stop_metric] * -1

                        if value < early_stop_best_value - self.options.early_stop_tolerance:
                            early_stop_progress = 0
                            early_stop_best_value = value
                        else:
                            early_stop_progress = early_stop_progress + 1

                        if early_stop_progress == self.options.early_stop_patience:
                            early_stop = True

                on_step_end_fn(step)
                if early_stop:
                    break
            on_epoch_end_fn(epoch)
            if early_stop:
                break
            toc = time.time()
            logger.info(f'-> Epoch@{epoch} Finished. Time Cost = {toc-tic:.2f}s')