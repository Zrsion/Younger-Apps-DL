#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-14 18:07:43
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Luzhou Peng (彭路洲) 
# Last Modified time: 2025-04-24 16:57:27
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import math
import tqdm
import torch
import bisect
import random
import pandas
import pathlib

from typing import Literal, Callable, Iterable
from pydantic import BaseModel, Field
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, top_k_accuracy_score

from younger.commons.io import create_dir

from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.engines import StandardTrainer, StandardTrainerOptions, StandardEvaluator, StandardEvaluatorOptions, StandardPredictor, StandardPredictorOptions, GraphSplit, GraphSplitOptions
from younger_apps_dl.datasets import NodeData, NodeDataset 
from younger_apps_dl.models import GAT_NP, GCN_NP, GIN_NP, SAGE_NP 

MODELS_MAP = {
    'GAT': GAT_NP,
    'GCN': GCN_NP,
    'GIN': GIN_NP,
    'SAGE': SAGE_NP
}

class ModelOptions(BaseModel):
    model_type: Literal['GAT', 'GCN', 'GIN', 'SAGE'] = Field('SAGE', description='The identifier of the model type, e.g., \'SAGE\', etc.')
    node_emb_dim: int = Field(512, description='Node embedding dimensionality.')
    hidden_dim: int = Field(256, description='Hidden layer dimensionality within the model.')
    dropout_rate: float = Field(0.5, description='Dropout probability used for regularization.')
    layer_number: int = Field(3, description='Number of layers (e.g., message-passing rounds for GNNs).')


class OptimizerOptions(BaseModel):
    lr: float = Field(0.001, description='Learning rate used by the optimizer.')
    eps: float = Field(1e-8, description='Epsilon for numerical stability.')
    weight_decay: float = Field(0.01, description='L2 regularization (weight decay) coefficient.')
    amsgrad: bool = Field(False, description='Whether to use the AMSGrad variant of the Adam optimizer.')


class SchedulerOptions(BaseModel):
    start_factor: float = Field(0.1, description='Initial learning rate multiplier for warm-up.')
    warmup_steps: int = Field(1500, description='Number of warm-up steps at the start of training.')
    total_steps: int = Field(150000, description='Total number of training steps for the scheduler to plan the learning rate schedule.')
    last_step: int = Field(-1, description='The last step index when resuming training. Use -1 to start fresh.')


class DatasetOptions(BaseModel):
    meta_filepath: pathlib.Path = Field(..., description='Path to the metadata file that describes the dataset.')
    raw_dirpath: pathlib.Path = Field(..., description='Directory containing raw input data files.')
    processed_dirpath: pathlib.Path = Field(..., description='Directory where processed dataset should be stored.')
    worker_number: int = Field(4, description='Number of workers for parallel data loading or processing.')


class BasicNodePredictionOptions(BaseModel):
    # Main Options
    logging_filepath: pathlib.Path | None = Field(None, description='Logging file path where logs will be saved, default to None, which may save to a default path that is determined by the Younger.')

    scheduled_sampling: bool = Field(False, description='Enable scheduled sampling during training to gradually shift from teacher forcing to model predictions.')
    scheduled_sampling_fixed: bool = Field(True, description='Use a fixed scheduled sampling ratio instead of dynamic scheduling.')
    scheduled_sampling_cycle: list[int] = Field([100], description='Training epochs at which to apply scheduled sampling updates in a cyclic manner.')
    scheduled_sampling_level: list[int] = Field([0], description='Sampling level (e.g., prediction depth or difficulty) applied at each cycle stage.')
    scheduled_sampling_ratio: float = Field(0.15, description='Initial probability of using model predictions instead of ground truth during training (between 0 and 1).')
    scheduled_sampling_micro: float = Field(12, description='Fine-grained control parameter for micro-level scheduled sampling behavior (e.g., per-step adjustment).')
    mask_ratio: float = Field(..., description='Ratio of elements (e.g., input tokens) to mask during training.')
    mask_method: Literal['Random', 'Purpose'] = Field(..., description='Strategy for masking elements: \'Random\' for uniform masking, \'Purpose\' for task-specific or guided masking.')

    trainer: StandardTrainerOptions
    evaluator: StandardEvaluatorOptions
    predictor: StandardPredictorOptions
    preprocessor: GraphSplitOptions

    train_dataset: DatasetOptions
    valid_dataset: DatasetOptions
    test_dataset: DatasetOptions

    model: ModelOptions
    optimizer: OptimizerOptions
    scheduler: SchedulerOptions


@register_task('ir', 'node_prediction')
class NodePrediction(BaseTask[BasicNodePredictionOptions]):
    OPTIONS = BasicNodePredictionOptions
    def train(self):
        self.valid_dataset = self._build_dataset_(
            self.options.valid_dataset.meta_filepath,
            self.options.valid_dataset.raw_dirpath,
            self.options.valid_dataset.processed_dirpath,
            'valid',
            self.options.valid_dataset.worker_number
        )
        self.train_dataset = self._build_dataset_(
            self.options.train_dataset.meta_filepath,
            self.options.train_dataset.raw_dirpath,
            self.options.train_dataset.processed_dirpath,
            'train',
            self.options.train_dataset.worker_number
        )
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.train_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.dropout_rate,
            self.options.model.layer_number
        )
        self.optimizer = self._build_optimizer_(
            self.model,
            self.options.optimizer.lr,
            self.options.optimizer.eps,
            self.options.optimizer.weight_decay,
            self.options.optimizer.amsgrad,
        )
        self.scheduler = self._build_scheduler_(
            self.optimizer,
            self.options.scheduler.start_factor,
            self.options.scheduler.warmup_steps,
            self.options.scheduler.total_steps,
            self.options.scheduler.last_step,
        )
        self.dicts = self.train_dataset.dicts

        trainer = StandardTrainer(self.options.trainer)
        trainer.run(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dataset,
            self.valid_dataset,
            self._train_fn_,
            self._valid_fn_,
            self._on_step_begin_fn_,
            self._on_step_end_fn_,
            self._on_epoch_begin_fn_,
            self._on_epoch_end_fn_,
            'pyg',
            self.options.logging_filepath
        )

    def evaluate(self):
        self.test_dataset = self._build_dataset_(
            self.options.test_dataset.meta_filepath,
            self.options.test_dataset.raw_dirpath,
            self.options.test_dataset.processed_dirpath,
            'test',
            self.options.test_dataset.worker_number
        )
        self.model = self._build_model_(
            len(self.test_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.dropout_rate,
            self.options.model.layer_number
        )
        self.dicts = self.test_dataset.dicts

        evaluator = StandardEvaluator(self.options.evaluator)
        evaluator.run(
            self.model,
            self.test_dataset,
            self._evaluate_fn_,
            'pyg',
            self.options.logging_filepath
        )

    def predict(self):
        predictor = StandardPredictor(self.options.predictor)
        self.dicts = NodeDataset.load_dicts(NodeDataset.load_meta(predictor.options.raw.load_dirpath.joinpath('meta.json')))
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.dropout_rate,
            self.options.model.layer_number
        )
        predictor.run(
            self.model,
            self._predict_raw_fn_,
            self.options.logging_filepath
        )

    def preprocess(self):
        preprocessor = GraphSplit(self.options.preprocessor)
        preprocessor.run(self.options.logging_filepath)

    def _build_model_(self, model_type: str, node_emb_size: int, node_emb_dim: int, hidden_dim: int, dropout_rate: float, layer_number: int) -> torch.nn.Module:
        model = MODELS_MAP[model_type](
            node_emb_size,
            node_emb_dim,
            hidden_dim,
            dropout_rate,
            layer_number
        )
        return model

    def _build_dataset_(self, meta_filepath: pathlib.Path, raw_dirpath: pathlib.Path, processed_dirpath: pathlib.Path, split: Literal['train', 'valid', 'test'], worker_number: int) -> NodeDataset:
        dataset = NodeDataset(
            meta_filepath,
            raw_dirpath,
            processed_dirpath,
            split=split,
            worker_number=worker_number
        )
        return dataset

    def _build_optimizer_(
        self,
        model: torch.nn.Module,
        lr: float,
        eps: float,
        weight_decay: float,
        amsgrad: float
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        return optimizer

    def _build_scheduler_(
        self,
        optimizer: torch.nn.Module,
        start_factor: float,
        warmup_steps: int,
        total_steps: int,
        last_step: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_lr_schr = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            total_iters=warmup_steps,
            last_epoch=last_step,
        )
        cosine_lr_schr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            last_epoch=last_step,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_schr, cosine_lr_schr],
            milestones=[warmup_steps],
            last_epoch=last_step,
        )
        return scheduler

    def _train_fn_(self, model: torch.nn.Module, minibatch: NodeData) -> list[tuple[str, torch.Tensor, Callable[[float], str]]]:
        device_descriptor = next(model.parameters()).device
        minibatch = minibatch.to(device_descriptor)
        x, edge_index, golden = self._mask_(minibatch, self.dicts['t2i'], self.options.mask_ratio, self.options.mask_method)

        if self.options.scheduled_sampling:
            scheduled_sampling_levels = list()
            # Only For -N, ..., -3, -2, -1
            for i in range(-self.scheduled_sampling_level_at_epoch, 0):
                if random.random() <= self.scheduled_sampling_ratio_at_epoch:
                    scheduled_sampling_levels.append(i)

            if len(scheduled_sampling_levels):
                model.eval()
                with torch.no_grad():
                    x, predict = self._simulate_predict_(model, minibatch, self.dicts['t2i'], scheduled_sampling_levels)

                model.train()
                x = x.detach()

        output = model(x.detach(), edge_index)
        loss = torch.nn.functional.cross_entropy(output, golden.squeeze(1), ignore_index=-1)
        loss.backward()
        return [('loss', loss, lambda x: f'{x:.4f}')]

    def _valid_fn_(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> list[tuple[str, torch.Tensor, Callable[[float], str]]]:
        device_descriptor = next(model.parameters()).device

        outputs = list()
        goldens = list()
        loss = 0
        # Return Output & Golden
        with tqdm.tqdm(total=len(dataloader)) as progress_bar:
            for index, minibatch in enumerate(dataloader, start=1):
                minibatch: NodeData = minibatch.to(device_descriptor)

                if self.options.scheduled_sampling:
                    x, edge_index, golden = self._mask_(minibatch, self.dicts['t2i'], 1, self.options.mask_method, test=True)
                    x, output = self._simulate_predict_(model, minibatch, self.dicts['t2i'], range(-self.options.scheduled_sampling_level, 0), test=True)
                else:
                    x, edge_index, golden = self._mask_(minibatch, self.dicts['t2i'], self.options.mask_ratio, self.options.mask_method)
                    output = torch.softmax(model(x, edge_index), dim=-1)
                loss += torch.nn.functional.cross_entropy(output, golden.squeeze(1), ignore_index=-1)

                outputs.append(output)
                goldens.append(golden)
                progress_bar.update(1)

        outputs = torch.cat(outputs)
        goldens = torch.cat(goldens).squeeze()

        val_indices = goldens != -1
        outputs = outputs[val_indices]
        goldens = goldens[val_indices]

        score = outputs.cpu().numpy()
        pred = outputs.max(1)[1].cpu().numpy()
        gold = goldens.cpu().numpy()

        print(f"pred[:5], pred[len(pred)//2:len(pred)//2+5]: {pred[:5]}, {pred[len(pred)//2:len(pred)//2+5]}")
        print(f"gold[:5], goldens[len(goldens)//2:len(goldens)//2+5]: {goldens[:5]}, {goldens[len(goldens)//2:len(goldens)//2+5]}")

        metrics = [
            ('loss', loss/len(dataloader), lambda x: f'{x:.4f}'),
            ('acc', torch.tensor(accuracy_score(gold, pred), device=x.device), lambda x: f'{x:.4f}'),
            ('macro_p', torch.tensor(precision_score(gold, pred, average='macro', zero_division=0), device=x.device), lambda x: f'{x:.4f}'),
            ('macro_r', torch.tensor(recall_score(gold, pred, average='macro', zero_division=0), device=x.device), lambda x: f'{x:.4f}'),
            ('macro_f1', torch.tensor(f1_score(gold, pred, average='macro', zero_division=0), device=x.device), lambda x: f'{x:.4f}'),
            ('micro_f1', torch.tensor(f1_score(gold, pred, average='micro', zero_division=0), device=x.device), lambda x: f'{x:.4f}'),
            ('top3_acc', torch.tensor(top_k_accuracy_score(gold, score, k = 3, labels=range(score.shape[1])), device=x.device), lambda x: f'{x:.4f}'),
            ('top5_acc', torch.tensor(top_k_accuracy_score(gold, score, k = 5, labels=range(score.shape[1])), device=x.device), lambda x: f'{x:.4f}') 
        ]
        return metrics

    def _evaluate_fn_(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        return self._valid_fn_(model, dataloader)

    def _on_step_begin_fn_(self, step: int) -> None:
        return

    def _on_step_end_fn_(self, step: int) -> None:
        return

    def _on_epoch_begin_fn_(self, epoch: int) -> None:
        ssc = self.options.scheduled_sampling_cycle
        assert all(ssc[i] < ssc[i+1] for i in range(len(ssc) - 1)), "Scheduled Sampling Cycle Must Be Strictly Increasing."
        i = min(bisect.bisect_left(ssc, epoch), len(ssc) - 1)
        self.scheduled_sampling_level_at_epoch = self.options.scheduled_sampling_level[i]
        if self.options.scheduled_sampling_fixed:
            r = self.options.scheduled_sampling_ratio
            self.scheduled_sampling_ratio_at_epoch = r
        else:
            m = self.options.scheduled_sampling_micro
            self.scheduled_sampling_ratio_at_epoch = 1 - m / (m + math.exp(epoch / m))
        return

    def _on_epoch_end_fn_(self, epoch: int) -> None:
        return

    def _mask_(self, minibatch: NodeData, t2i: dict[str, int], mask_ratio: float, mask_method: Literal['Random', 'Purpose'], test: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device_descriptor = minibatch.x.device
        x = minibatch.x.clone()
        edge_index = minibatch.edge_index.clone()

        golden = x.clone()

        if mask_method == 'Random':
            mask_probability = torch.full(x.shape, mask_ratio, dtype=torch.float, device=device_descriptor)

        if mask_method == 'Purpose':
            source_index = edge_index[0]
            target_index = edge_index[1]

            last_level_nodes = target_index[~torch.isin(target_index, source_index)]

            unique = torch.unique(last_level_nodes)
            if unique.shape[0] != 0:
                mask_ratio = mask_ratio * x.shape[0] / unique.shape[0]
                mask_ratio = 1 if mask_ratio > 1 else mask_ratio

            mask_probability = torch.zeros_like(x, dtype=torch.float, device=self.device_descriptor)
            mask_probability[last_level_nodes] = torch.full(x.shape, mask_ratio, dtype=torch.float, device=device_descriptor)[last_level_nodes]

        mask_indices = torch.bernoulli(mask_probability).to(device_descriptor).bool()
        golden[~mask_indices] = -1

        if test:
            mask_mask_indices = torch.bernoulli(torch.full(x.shape, 1.0, dtype=torch.float, device=device_descriptor)).bool() & mask_indices
            x[mask_mask_indices] = t2i['__MASK__']
            # x = torch.where(mask_indices, torch.full_like(x, t2i['__MASK__']), x)
        else:
            mask_mask_indices = torch.bernoulli(torch.full(x.shape, 0.8, dtype=torch.float, device=device_descriptor)).bool() & mask_indices
            x[mask_mask_indices] = t2i['__MASK__']
            # x = torch.where(mask_indices, torch.full_like(x, t2i['__MASK__']), x)

            mask_optr_indices = torch.bernoulli(torch.full(x.shape, 0.5, dtype=torch.float, device=device_descriptor)).bool() & mask_indices & ~mask_mask_indices
            x[mask_optr_indices] = torch.randint(2, len(t2i), x.shape, dtype=torch.long, device=device_descriptor)[mask_optr_indices]
            # rand_values = torch.randint(2, len(t2i), x.shape, dtype=torch.long, device=device_descriptor)
            # x = torch.where(mask_optr_indices, rand_values, x)

        return x, edge_index, golden

    def _simulate_predict_(self, model: torch.nn.Module, minibatch: NodeData, t2i: dict[str, int], simulate_levels: Iterable[int], test: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        device_descriptor = minibatch.x.device
        x = minibatch.x.clone()
        edge_index = minibatch.edge_index
        predict = torch.zeros_like(model(x, edge_index))
        for predict_level in simulate_levels:
            level = minibatch.level
            predict_nodes = torch.where(level == predict_level)[0]
            changed_nodes = torch.where(level <= predict_level)[0]
            old2new = torch.zeros(x.shape[0], dtype=torch.long).to(device_descriptor)
            old2new[changed_nodes] = torch.arange(changed_nodes.shape[0], device=device_descriptor)

            node_mask = torch.zeros(x.shape[0], dtype=torch.bool).to(device_descriptor)
            node_mask[changed_nodes] = True

            
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            changed_edge_index = old2new[edge_index[:, edge_mask]]

            changed_x = x[changed_nodes]
            changed_x[old2new[predict_nodes]] = t2i['__MASK__']
            changed_predict = torch.softmax(model(changed_x, changed_edge_index), dim=-1)

            if test:
                output = torch.argmax(changed_predict, dim=-1, keepdim=True)
            else:
                output = torch.argmax(torch.multinomial(changed_predict, num_samples=1), dim = -1, keepdim=True)

            x[predict_nodes] = output[old2new[predict_nodes]]
            predict[predict_nodes] = changed_predict[old2new[predict_nodes]]
            return x, predict

    def _predict_raw_fn_(self, model: torch.nn.Module, load_dirpath: pathlib.Path, save_dirpath: pathlib.Path):
        logicx_filepaths = [logicx_filepath for logicx_filepath in load_dirpath.joinpath('logicxs').iterdir()]
        device_descriptor = next(model.parameters()).device

        from torch_geometric.loader import NeighborLoader
        from younger_logics_ir.modules import LogicX

        create_dir(save_dirpath)

        graph_hashes = list()
        graph_embeddings = list()
        for logicx_filepath in logicx_filepaths:
            logicx = LogicX()
            logicx.load(logicx_filepath)
            graph_hashes.append(LogicX.hash(logicx))

            data = NodeDataset.process_graph_data(logicx, self.dicts)
            loader = NeighborLoader(
                data,
                num_neighbors=[-1] * len(model.encoder.layers),
                batch_size=512,
                input_nodes=None,
                subgraph_type="directional",
                directed=True
            )

            embedding_dim = model.encoder.node_.embedding_layer.embedding_dim
            graph_embedding = torch.zeros(embedding_dim, device=device_descriptor)
            node_count = 0
            for batch in loader:
                batch: NodeData = batch.to(device_descriptor)
                out = model.encoder(batch.x, batch.edge_index)               # shape: [total_nodes_in_batch, dim]
                center_embeddings = out[:batch.batch_size]                   # shape: [batch_size, dim]
                graph_embedding += center_embeddings.sum(dim=0)
                node_count += center_embeddings.shape[0]
            assert len(logicx.dag) == node_count
            graph_embedding = (graph_embedding/node_count).detach().cpu().numpy().tolist()
            graph_embeddings.append(graph_embedding)

        hsh_df = pandas.DataFrame(graph_hashes, columns=["logicx_hash"])
        hsh_df.to_csv(save_dirpath.joinpath("graph_hashes.csv"), index=False)

        emb_df = pandas.DataFrame(graph_embeddings, columns=[str(i) for i in range(embedding_dim)])
        emb_df.to_csv(save_dirpath.joinpath("graph_embeddings.csv"), index=False)
