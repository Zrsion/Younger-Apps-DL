#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
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


class GraphSplitOptions(BaseModel):
    load_dirpath: pathlib.Path = Field(..., description='Directory path to load LogicX\'s.')
    save_dirpath: pathlib.Path = Field(..., description='Directory path to save LogicX\'s.')

    method: Literal['Random', 'Cascade', 'RandomFull', 'CascadeFull', 'Window', 'MixBasic', 'MixSuper'] = Field(..., description='Graph splitting method. '
                                                                                               '\'Random\' selects a random node as center; BFS is used to expand the subgraph, retaining a random subset of nodes at each depth. '
                                                                                               '\'RandomFull\' is similar, but retains all nodes at each BFS depth. '
                                                                                               '\'Cascade\' restricts the expansion to ancestors or descendants of the center node, retaining a random subset at each depth. '
                                                                                               '\'CascadeFull\' is the full-retention version of \'Cascade\', preserving all nodes at each BFS depth.'
                                                                                               '\'Window\' randomly selects a graph and identifies nodes at a specific level; then performs a `split_scale`-step backward traversal, incorporating all traversed nodes and edges into the subgraph.'
                                                                                               '\'MixBasic\' uniformly samples one of the following methods for each subgraph: \'Random\', \'RandomFull\', \'Cascade\', or \'CascadeFull\'. '
                                                                                               '\'MixSuper\' extends MixBasic by additionally including \'Window\' in the set of candidate methods, sampling uniformly among all five.')

    split_scale: list[int] = Field(..., description='List of node counts to include in subgraph splits expanded from central nodes. Each value specifies a different subgraph split scale to generate.')
    split_count: int = Field(..., description='Number of subgraph splits to generate per central node.')
    split_tries: int = Field(..., description='Maximum number of attempts to generate `split_count` valid subgraphs (e.g., avoiding duplicates or undersized splits).')
    split_limit: int = Field(..., description='Maximum allowed size (in number of nodes) for a subgraph split. If a candidate subgraph exceeds this size, it will be discarded. '
                                                    'This limit only applies to methods that involve the \'Window\' extraction strategy, including \'Window\' and mixed strategies \'MixSuper\'.')

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


@register_engine('preprocessor', 'graph_split')
class GraphSplit(BaseEngine[GraphSplitOptions]):
    OPTIONS = GraphSplitOptions

    def run(
        self,
        logging_filepath: pathlib.Path | None = None,
    ) -> None:
        equip_logger(logging_filepath)

        random.seed(self.options.seed)
        numpy.random.seed(self.options.seed)

        logger.info(f'')

        logicx_filepaths = sorted([logicx_filepath for logicx_filepath in self.options.load_dirpath.iterdir()])

        uuid2tuid = dict() # {uuid: tuid}

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
                    tuid = logicx.dag.nodes[node_index]['node_tuid']
                    uuid2tuid[uuid] = tuid
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

        logger.info(f'User Specified # of Splits - Training/Validation/Test = {self.options.training_dataset_size} / {self.options.validation_dataset_size} / {self.options.test_dataset_size}')

        expected_overall_split_count = len(uuid_occurence)*len(self.options.split_scale)*min(self.options.split_count, self.options.split_tries)
        expected_training_dataset_size = min(expected_overall_split_count, self.options.training_dataset_size)
        expected_validation_dataset_size = min(max(0, expected_overall_split_count-self.options.training_dataset_size), self.options.validation_dataset_size)
        expected_test_dataset_size = min(max(0, expected_overall_split_count-self.options.training_dataset_size-self.options.validation_dataset_size), self.options.test_dataset_size)

        logger.info(f'split_tries = {self.options.split_tries}')
        logger.info(f'split_count = {self.options.split_count}')
        logger.info(f'Expected Overal # of Splits = {expected_overall_split_count}')
        logger.info(f'Expected # of Splits - Training/Validation/Test = {expected_training_dataset_size} / {expected_validation_dataset_size} / {expected_test_dataset_size}')

        logger.info(f'Spliting ...')
        splits: dict[int, dict[str, LogicX]] = {split_scale: dict() for split_scale in self.options.split_scale} # {split_scale: {split_hash: split}}
        split_hashes: dict[int, dict[str, list[str]]] = {split_scale: dict() for split_scale in self.options.split_scale} # {split_scale: {uuid: list[split_hash]}}
        # For Each Split Size:
        for split_scale in self.options.split_scale:
            # For Each Operator:
            logger.info(f' -> Now Retrieving Subgraph Splits with Size {split_scale}...')
            with tqdm.tqdm(total=len(all_uuid_positions)) as progress_bar:
                for uuid, uuid_positions in all_uuid_positions.items():

                    current_tries = 0
                    current_split_count = 0
                    candidate_logicx_indices: set[int] = set(uuid_positions.keys())
                    # Generate Subgraph Split Repeatedly
                    while len(candidate_logicx_indices) != 0 and current_split_count < self.options.split_count:
                        if not (current_tries < self.options.split_tries):
                            break
                        selected_logicx_index: int = int(numpy.random.choice(list(candidate_logicx_indices)))
                        selected_node_index: str = str(numpy.random.choice(list(uuid_positions[selected_logicx_index])))

                        method = self.options.method
                        if self.options.method == 'MixBasic':
                            method = random.choice(['Random', 'Cascade', 'RandomFull', 'CascadeFull'])
                        if self.options.method == 'MixSuper':
                            method = random.choice(['Random', 'Cascade', 'RandomFull', 'CascadeFull', 'Window'])

                        if method == 'Window':
                            selected_node_order: int = all_nid2nod[selected_logicx_index][selected_node_index]
                            if selected_node_order  < split_scale - 1:
                                continue
                            selected_node_indices: list[int] = all_nod2nids[selected_logicx_index][selected_node_order]
                            split = self.__class__.retrieve_split(logicxs[selected_logicx_index], selected_node_indices, split_scale, self.options.split_limit, self.options.method)
                            split_size = split_scale
                        else:
                            split = self.__class__.retrieve_split(logicxs[selected_logicx_index], [selected_node_index], split_scale, self.options.split_limit, self.options.method)
                            split_size = len(split.dag)

                        current_tries += 1
                        if split_size not in self.options.split_scale:
                            continue
                        split_hash = LogicX.hash(split)
                        if split_hash in splits[split_size]:
                            continue
                        split.dag.graph['origin'] = logicx_hashes[selected_logicx_index]
                        splits[split_size][split_hash] = split
                        current_split_count += 1
                        # Add To Split
                        split_hashes[split_size].setdefault(uuid, list()).append(split_hash)
                    if current_split_count < self.options.split_count:
                        flag = f'No!'
                    else:
                        flag = f'Ye!'
                    progress_bar.set_description(f'Current: {uuid} Enough: ({flag})')
                    progress_bar.update(1)

        split_with_hashes = [
            (split_hash, splits[split_scale][split_hash])
            for split_scale, split_hashes_at_split_scale in split_hashes.items()
            for uuid, uuid_split_hashes_at_split_scale in split_hashes_at_split_scale.items()
            for index, split_hash in enumerate(uuid_split_hashes_at_split_scale)
        ]

        random.shuffle(split_with_hashes)

        exact_training_dataset_size = min(self.options.training_dataset_size, len(split_with_hashes))
        exact_validation_dataset_size = min(self.options.validation_dataset_size, len(split_with_hashes)-exact_training_dataset_size)
        exact_test_dataset_size = min(self.options.test_dataset_size, len(split_with_hashes)-exact_training_dataset_size-exact_validation_dataset_size)

        logger.info(f'Exact # of Splits - Training/Validation/Test = {exact_training_dataset_size} / {exact_validation_dataset_size} / {exact_test_dataset_size}')

        training_dataset_save_dirpath = self.options.save_dirpath.joinpath('training')
        logger.info(f'Saving \'Training\' Dataset into {training_dataset_save_dirpath.absolute()} ... ')
        self.__class__.save_dataset(uuid_occurence, split_with_hashes[:exact_training_dataset_size], training_dataset_save_dirpath, ignored, uuid2tuid)

        validation_dataset_save_dirpath = self.options.save_dirpath.joinpath('validation')
        logger.info(f'Saving \'Validation\' Dataset into {validation_dataset_save_dirpath.absolute()} ... ')
        self.__class__.save_dataset(uuid_occurence, split_with_hashes[exact_training_dataset_size:exact_training_dataset_size+exact_validation_dataset_size], validation_dataset_save_dirpath, ignored, uuid2tuid)

        test_dataset_save_dirpath = self.options.save_dirpath.joinpath('test')
        logger.info(f'Saving \'Test\' Dataset into {test_dataset_save_dirpath.absolute()} ... ')
        self.__class__.save_dataset(uuid_occurence, split_with_hashes[exact_training_dataset_size+exact_validation_dataset_size:exact_training_dataset_size+exact_validation_dataset_size+exact_test_dataset_size], test_dataset_save_dirpath, ignored, uuid2tuid)

    @classmethod
    def retrieve_split(cls, logicx: LogicX, center_node_indices: list[str], split_scale: int, split_limit: int, method: Literal['Random', 'Cascade', 'RandomFull', 'CascadeFull', 'Window']) -> LogicX:
        # Direction: Literal[0, 1, -1] Center Node: 0; Successor: 1; Predecessors: -1;
        bfs_flags = set(center_node_indices)
        bfs_queue = collections.deque([[center_node_index, 0] for center_node_index in center_node_indices])
        if method == 'Window':
            current_level = 0
            while len(bfs_queue) != 0 and current_level > -split_scale:
                prev_level_size = len(bfs_queue)
                for _ in range(prev_level_size):
                    node_index = bfs_queue.popleft()
                    neighbors = numpy.random.shuffle([predecessor for predecessor in logicx.dag.predecessors(node_index)])
                    for neighbor in neighbors:
                        if len(bfs_flags) < split_limit and predecessor not in bfs_flags:
                            bfs_flags.add(predecessor)
                            bfs_queue.append(predecessor)
                current_level = current_level - 1
        else:
            while len(bfs_queue) != 0 and len(bfs_flags) < split_scale:
                current_node_index, direction = bfs_queue.popleft()
                next_levels = list()

                if method in ['Random', 'RandomFull']:
                    for neighbor in networkx.function.all_neighbors(logicx.dag, current_node_index):
                        next_levels.append((neighbor, 0))

                if method in ['Cascade', 'CascadeFull']:
                    if 0 <= direction:
                        for neighbor in logicx.dag.successors(current_node_index):
                            next_levels.append((neighbor, 1))

                    if direction <= 0:
                        for neighbor in logicx.dag.predecessors(current_node_index):
                            next_levels.append((neighbor, -1))

                if len(next_levels) == 0:
                    continue

                numpy.random.shuffle(next_levels)
                if method in ['Random', 'Cascade']:
                    limit = numpy.random.randint(1, len(next_levels) + 1)
                    for neighbor, direction in next_levels[:limit]:
                        if len(bfs_flags) < split_scale and neighbor not in bfs_flags:
                            bfs_flags.add(neighbor)
                            bfs_queue.append([neighbor, direction])

                if method in ['RandomFull', 'CascadeFull']:
                    if len(bfs_flags) < split_scale:
                        for neighbor, direction in next_levels:
                            if neighbor not in bfs_flags:
                                bfs_flags.add(neighbor)
                                bfs_queue.append([neighbor, direction])

        induced_subgraph = logicx.dag.subgraph(bfs_flags)

        subgraph = networkx.DiGraph()
        subgraph.graph['level'] = True
        subgraph.add_nodes_from(induced_subgraph.nodes(data=False))
        subgraph.add_edges_from(induced_subgraph.edges(data=False))
        for node_index in subgraph.nodes():
            subgraph.nodes[node_index]['node_uuid'] = induced_subgraph.nodes[node_index]['node_uuid']
            subgraph.nodes[node_index]['node_indegree'] = subgraph.in_degree(node_index)
            subgraph.nodes[node_index]['node_outdegree'] = subgraph.out_degree(node_index)
        node_ods: dict[str, int] = {node_index: subgraph.out_degree(node_index) for node_index in subgraph.nodes}
        bfs_queue = collections.deque()
        for node_index in subgraph.nodes:
            if node_ods[node_index] == 0:
                bfs_queue.append(node_index)
        while len(bfs_queue) != 0:
            prev_level_size = len(bfs_queue)
            for _ in range(prev_level_size):
                node_index = bfs_queue.popleft()
                subgraph.nodes[node_index]['level'] = min([subgraph.nodes[successor].get('level', 0) - 1 for successor in subgraph.successors(node_index)] + [0])
                for predecessor in subgraph.predecessors(node_index):
                    node_ods[predecessor] -= 1
                    if node_ods[predecessor] == 0:
                        bfs_queue.append(predecessor)

        split = LogicX(src=logicx.src, dag=subgraph)
        return split

    @classmethod
    def save_dataset(cls, uuid_occurence: dict[str, int], split_with_hashes: list[tuple[str, LogicX]], save_dirpath: pathlib.Path, ignored: set[str], uuid2tuid):
        node_types = [node_type for node_type, node_occr in uuid_occurence.items() if node_type not in ignored]
        item_names = [item_name for item_name, item_lgcx in split_with_hashes]
        meta = dict(
            node_types = node_types,
            item_names = item_names,
            uuid2tuid = uuid2tuid,
        )

        items_dirpath = save_dirpath.joinpath('items')
        create_dir(items_dirpath)
        meta_filepath = save_dirpath.joinpath('meta.json')

        logger.info(f'Saving META ... ')
        save_json(meta, meta_filepath, indent=2)
        logger.info(f'Saved.')

        logger.info(f'Saving Items ... ')
        with tqdm.tqdm(total=len(split_with_hashes), desc='Saving') as progress_bar:
            for split_hash, split in split_with_hashes:
                item_filepath = items_dirpath.joinpath(f'{split_hash}')
                split.save(item_filepath)
                progress_bar.update(1)
        logger.info(f'Saved.')
