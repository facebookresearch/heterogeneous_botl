#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from heterogeneous_botl.benchmarks.hpo_b.hpob_handler import HPOBHandler


def get_train_data_by_search_id(
    hpob_hdlr: HPOBHandler,
    search_space_id: str,
    n_source_samples: int,
    dataset_index: int = 0,
):
    """
    Get the training data for a search space id. HPO-B benchmark has a
    set of training data and test data for each search space id.
    Each search space id has multiple datasets. This function returns
    data from one of the dataset with more than n_source_samples samples.
    """
    dataset_keys = list(hpob_hdlr.meta_train_data[search_space_id].keys())
    print(f"No of training dataset_keys {len(dataset_keys)}")
    source_keys = [
        key
        for key in dataset_keys
        if len(hpob_hdlr.meta_train_data[search_space_id][key]["X"]) > n_source_samples
    ]
    key = source_keys[dataset_index]
    X = hpob_hdlr.meta_train_data[search_space_id][key]["X"]
    y = hpob_hdlr.meta_train_data[search_space_id][key]["y"]
    return X, y


def get_test_data_by_search_id(
    hpob_hdlr: HPOBHandler, search_space_id: str, dataset_index: int = 0
):
    dataset_keys = list(hpob_hdlr.meta_test_data[search_space_id].keys())
    print(f"No of test dataset_keys {len(dataset_keys)}")
    key = dataset_keys[dataset_index]
    X = hpob_hdlr.meta_test_data[search_space_id][key]["X"]
    y = hpob_hdlr.meta_test_data[search_space_id][key]["y"]
    return X, y
