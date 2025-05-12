#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Luzhou Peng (彭路洲)
# Last Modified time: 2025-04-24 09:48:02
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import numpy
import random
import pathlib
import networkx
import collections

from typing import Any, Literal
from pydantic import BaseModel, Field

from younger.commons.io import save_json, create_dir

from younger_logics_ir.modules import LogicX

from younger_apps_dl.commons.logging import logger, equip_logger

from younger_apps_dl.engines import BaseEngine, register_engine


class EdgeDatasetSplitOptions(BaseModel):
    load_dirpath: pathlib.Path = Field(..., description='Directory path to load LogicX\'s.')
    save_dirpath: pathlib.Path = Field(..., description='Directory path to save LogicX\'s.')

    training_dataset_size: int = Field(..., description='Number of subgraphs splits to include in the training set.')
    validation_dataset_size: int = Field(..., description='Number of subgraphs splits to include in the validation set.')
    test_dataset_size: int = Field(..., description='Number of subgraph splits to include in the test set.')

    min_graph_size: int | None = Field(None, ge=0, description='Minimum number of nodes a full graph must have to be considered for graph split. '
                                                               'Graphs smaller than this value will be excluded. '
                                                               'Set to `None` to disable this filter.')
    max_graph_size: int | None = Field(None, ge=0, description='Maximum number of nodes a full graph must have to be considered for graph split. '
                                                               'Graphs larger than this value will be excluded. '
                                                               'Set to `None` to disable this filter.')

    uuid_threshold: int | None = Field(None, ge=0, description='Occurence threshold to ignore uuid, lower than threshold will be discarded.')
    seed: int = Field(16861, ge=0, description='Random seed for deterministic behavior during subgraph split sampling.')


@register_engine('preprocessor', 'edge_dataset_split')
class EdgeDatasetSplit(BaseEngine[EdgeDatasetSplitOptions]):
    OPTIONS = EdgeDatasetSplitOptions

    def run(
        self,
        logging_filepath: pathlib.Path | None = None,
    ) -> None:
        equip_logger(logging_filepath)

        random.seed(self.options.seed)
        numpy.random.seed(self.options.seed)

        logger.info(f'')

        logicx_filepaths = sorted([logicx_filepath for logicx_filepath in self.options.load_dirpath.iterdir()])

        logger.info(f'Scan Load Directory & Generate Node UUID List ...')
        logicxs: list[LogicX] = list() # [logicx1, logicx2, ...]
        logicx_hashes: list[str] = list() # [logicx1_hash, logicx2_hash, ...]
        all_uuid_positions: dict[str, dict[int, set[str]]] = dict() # {uuid: {logicx_index: set[node_index]}}
        all_nid2nod: dict[str, dict[str, int]] = dict() # {logicx_index: {node_index: order}}
        all_nod2nids: dict[str, dict[int, list[str]]] = dict() # {logicx_index: {order: list[node_index]}}

        with tqdm.tqdm(total=len(logicx_filepaths)) as progress_bar:
            logicx_index = 0
            for logicx_filepath in logicx_filepaths:
                progress_bar.update(1)
                logicx = LogicX()
                logicx.load(logicx_filepath)

                graph_size = len(logicx.dag)
                if self.options.min_graph_size is None or self.options.min_graph_size <= graph_size:
                    min_graph_size_meet = True
                else:
                    min_graph_size_meet = False

                if self.options.max_graph_size is None or graph_size <= self.options.max_graph_size:
                    max_graph_size_meet = True
                else:
                    max_graph_size_meet = False
                if not (min_graph_size_meet and max_graph_size_meet):
                    continue

                logicxs.append(logicx)
                logicx_hashes.append(logicx_filepath.name)

                for node_index in logicx.dag.nodes:
                    uuid = logicx.dag.nodes[node_index]['node_uuid']

                    uuid_positions = all_uuid_positions.get(uuid, dict())
                    node_indices = uuid_positions.get(logicx_index, set())
                    node_indices.add(node_index)
                    uuid_positions[logicx_index] = node_indices
                    all_uuid_positions[uuid] = uuid_positions

                all_nid2nod[logicx_index] = dict()
                all_nod2nids[logicx_index] = dict()
                for node_index in networkx.topological_sort(logicx.dag):
                    predecessors = logicx.dag.predecessors(node_index)
                    all_nid2nod[logicx_index][node_index] = max([all_nid2nod[logicx_index][predecessor] + 1 for predecessor in predecessors] + [0])
                    all_nod2nids[logicx_index].setdefault(all_nid2nod[logicx_index][node_index], list()).append(node_index)

                logicx_index += 1

        uuid_occurence: dict[str, int] = dict()
        for uuid, uuid_positions in all_uuid_positions.items():
            uuid_occurence[uuid] = sum([len(node_indices) for logicx_index, node_indices in uuid_positions.items()])
        logger.info(f'Total {len(uuid_occurence)} Different Operators')

        ignored = set(uuid for uuid, occurence in uuid_occurence.items() if self.options.uuid_threshold is not None and self.options.uuid_threshold <= occurence )
        logger.info(f'After Ignore: {len(uuid_occurence) - len(ignored)} Different Operators')

        logger.info(f'Spliting ...')
        # For Each Operator:
        logicxs_with_hashes: dict[str, list[tuple[str, LogicX]]] = dict() # {uuid: [(logicx_hash, logicx), ...]} 
        logicxs_with_hashes =  [(logicx_hash, logicx) for logicx_hash, logicx in zip(logicx_hashes, logicxs)]
        # split_with_hashes = [
        #     (split_hash, splits[split_scale][split_hash])
        #     for split_scale, split_hashes_at_split_scale in split_hashes.items()
        #     for uuid, uuid_split_hashes_at_split_scale in split_hashes_at_split_scale.items()
        #     for index, split_hash in enumerate(uuid_split_hashes_at_split_scale)
        # ]
        
        random.shuffle(logicxs_with_hashes)
        expected_total_size = self.options.training_dataset_size + self.options.validation_dataset_size + self.options.test_dataset_size

        exact_training_dataset_size = min(self.options.training_dataset_size, round(len(logicxs_with_hashes) * (self.options.training_dataset_size / expected_total_size)))
        exact_validation_dataset_size = min(self.options.validation_dataset_size, round(len(logicxs_with_hashes) * (self.options.validation_dataset_size / expected_total_size)))
        exact_test_dataset_size = len(logicxs_with_hashes) - exact_training_dataset_size - exact_validation_dataset_size

        logger.info(f'Exact # of Splits - Training/Validation/Test = {exact_training_dataset_size} / {exact_validation_dataset_size} / {exact_test_dataset_size}')

        training_dataset_save_dirpath = self.options.save_dirpath.joinpath('training')
        logger.info(f'Saving \'Training\' Dataset into {training_dataset_save_dirpath.absolute()} ... ')
        self.__class__.save_dataset(uuid_occurence, logicxs_with_hashes[:exact_training_dataset_size], training_dataset_save_dirpath, ignored)

        validation_dataset_save_dirpath = self.options.save_dirpath.joinpath('validation')
        logger.info(f'Saving \'Validation\' Dataset into {validation_dataset_save_dirpath.absolute()} ... ')
        self.__class__.save_dataset(uuid_occurence, logicxs_with_hashes[exact_training_dataset_size:exact_training_dataset_size+exact_validation_dataset_size], validation_dataset_save_dirpath, ignored)

        test_dataset_save_dirpath = self.options.save_dirpath.joinpath('test')
        logger.info(f'Saving \'Test\' Dataset into {test_dataset_save_dirpath.absolute()} ... ')
        self.__class__.save_dataset(uuid_occurence, logicxs_with_hashes[exact_training_dataset_size+exact_validation_dataset_size:exact_training_dataset_size+exact_validation_dataset_size+exact_test_dataset_size], test_dataset_save_dirpath, ignored)


    @classmethod
    def save_dataset(cls, uuid_occurence: dict[str, int], logicxs_with_hashes: list[tuple[str, LogicX]], save_dirpath: pathlib.Path, ignored: set[str]):
        node_types = [node_type for node_type, node_occr in uuid_occurence.items() if node_type not in ignored]
        item_names = [item_name for item_name, item_lgcx in logicxs_with_hashes]
        meta = dict(
            node_types = node_types,
            item_names = item_names,
        )

        items_dirpath = save_dirpath.joinpath('items')
        create_dir(items_dirpath)
        meta_filepath = save_dirpath.joinpath('meta.json')

        logger.info(f'Saving META ... ')
        save_json(meta, meta_filepath, indent=2)
        logger.info(f'Saved.')

        logger.info(f'Saving Items ... ')
        with tqdm.tqdm(total=len(logicxs_with_hashes), desc='Saving') as progress_bar:
            for logicx_hash, logicx in logicxs_with_hashes:
                item_filepath = items_dirpath.joinpath(f'{logicx_hash}')
                logicx.save(item_filepath)
                progress_bar.update(1)
        logger.info(f'Saved.')
