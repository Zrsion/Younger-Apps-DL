#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-21 12:24
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import tqdm
import pathlib
import torch
import random
import networkx
import multiprocessing

from typing import Any, Callable, Literal
from torch_geometric.io import fs
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import is_sparse

from younger.commons.io import load_json, load_pickle, tar_extract
from younger.commons.utils import split_sequence
from younger_logics_ir.modules import LogicX

from younger_apps_dl.datasets import register_dataset


@register_dataset('node')
class NodeData(Data):
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:

        if is_sparse(value) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:

        if 'batch' in key and isinstance(value, torch.Tensor):
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0


class NodeDataset(Dataset):
    def __init__(
        self,

        meta_filepath: str,
        raw_dirpath: str,
        processed_dirpath: str,
        split: Literal['train', 'valid', 'test'],
        worker_number: int = 4,
        all_in: bool = True,

        root: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,
    ):
        self.worker_number = worker_number
        self.split = split
        self.all_in = all_in

        self.meta_filepath = meta_filepath
        self.raw_dirpath = raw_dirpath
        self.processed_dirpath = processed_dirpath
        
        self.meta = self.__class__.load_meta(self.meta_filepath)
        self.hashs = self.__class__.load_hashs(self.meta)
        self.dicts = self.__class__.load_dicts(self.meta)

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

        self.all_graph_data: list[NodeData] = list()
        if self.all_in:
            for processed_path in tqdm.tqdm(self.processed_paths, total=len(self.processed_paths), desc="All In GraphData:"):
                self.all_graph_data.append(torch.load(processed_path))
                
    @property
    def raw_file_names(self):
        return [f'{hash}' for hash in self.hashs]

    @property
    def processed_file_names(self):
        return [f'{hash}' for hash in self.hashs]

    @property
    def raw_dir(self) -> str:
        return self.raw_dirpath

    @property
    def processed_dir(self) -> str:
        return self.processed_dirpath

    def len(self) -> int:
        return len(self.hashs)

    def get(self, index: int) -> NodeData:
        if self.all_in:
            graph_data = self.all_graph_data[index]
        else:
            graph_data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[index]))
        return graph_data
    
    def _process_chunk_(self, parameter: tuple[list[int], int]):
        indices, worker_id = parameter
        processed_indices = list()
        with tqdm.tqdm(total=len(indices), desc=f"Processing: Worker PID - {os.getpid()}", position=worker_id) as progress_bar:
            for index in indices:
                self.process_sample(index)
                processed_indices.append(index)
                progress_bar.set_postfix({f'Current Indices': f'{index}'})
                progress_bar.update(1)
        return processed_indices

    def process(self):
        chunk_count = self.worker_number * 4
        indices_list: list[list[int]] = split_sequence(list(range(len(self))), chunk_count)
        worker_index: list[int] = list(range(self.worker_number)) * 4
        with multiprocessing.Pool(self.worker_number) as pool:
            for indices in pool.imap_unordered(self._process_chunk_, zip(indices_list, worker_index)):
                pass

    def process_sample(self, index: int) -> int:
        logicx_filepath = os.path.join(self.raw_dir, self.raw_file_names[index])
        logicx = LogicX()
        logicx.load(pathlib.Path(logicx_filepath))
        graph_data = self.__class__.process_node_data(logicx, self.dicts)
        torch.save(graph_data, os.path.join(self.processed_dir, self.processed_file_names[index]))
        return index
    
    @classmethod
    def load_meta(cls, meta_filepath: str) -> dict[str, Any]:
        meta: dict[str, Any] = load_json(meta_filepath)
        return meta

    @classmethod
    def load_hashs(cls, meta: dict[str, Any]) -> list[str]:
        assert 'item_names' in meta
        assert isinstance(meta['item_names'], list)
        hashs = sorted(meta['item_names'])
        return hashs

    @classmethod
    def load_dicts(cls, meta: dict[str, Any]) -> dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]]:
        assert 'node_types' in meta
        assert isinstance(meta['node_types'], list)

        dicts: dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]] = dict()
        dicts['i2t'] = dict()
        dicts['t2i'] = dict()

        node_types = ['__UNK__'] + ['__MASK__'] + meta['node_types']
        for i, t in enumerate(node_types):
            dicts['i2t'][i] = t
            dicts['t2i'][t] = i
        return dicts
    

    @classmethod
    def process_graph_x(cls, logicx: LogicX, dicts: dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]], nxid2pgid: dict[str, int]) -> torch.Tensor:
        # Shape: [#Node, 1]

        # ID in DAG
        node_indices_in_dag: list[str] = sorted(list(logicx.dag.nodes), key=lambda x: nxid2pgid[x])

        # ID in Dict
        node_indices_in_dict = list()
        for node_index_in_dag in node_indices_in_dag:
            node_uuid = logicx.dag.nodes[node_index_in_dag]['node_uuid']
            if node_uuid in dicts['t2i']:
                node_index_in_dict = [dicts['t2i'][node_uuid]]
            else:
                node_index_in_dict = [dicts['t2i']['__UNK__']]
            node_indices_in_dict.append(node_index_in_dict)
        x = torch.tensor(node_indices_in_dict, dtype=torch.long)

        return x

    @classmethod
    def process_graph_edge_index(cls, logicx: LogicX, nxid2pgid: dict[str, int]) -> torch.Tensor:
        # Shape: [2, #Edge]
        edge_index = torch.empty((2, logicx.dag.number_of_edges()), dtype=torch.long)
        for index, (src, dst) in enumerate(list(logicx.dag.edges)):
            edge_index[0, index] = nxid2pgid[src]
            edge_index[1, index] = nxid2pgid[dst]
        return edge_index

    @classmethod
    def process_node_data(
        cls,
        logicx: LogicX,
        dicts: dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]],
    ) -> NodeData:
        nxids = sorted(list(logicx.dag.nodes))
        pgids = list(range(logicx.dag.number_of_nodes()))
        nxid2pgid = dict(zip(nxids, pgids))
        # >>> print(list(sorted(logicx.dag.nodes)))
        # [B, A, C, D]
        # >>> print(nxid2pgid)
        # {A: 0, B: 1, C: 2, D: 3}

        x = cls.process_graph_x(logicx, dicts, nxid2pgid)
        edge_index = cls.process_graph_edge_index(logicx, nxid2pgid)
        node_data = NodeData(x=x, edge_index=edge_index)
        return node_data