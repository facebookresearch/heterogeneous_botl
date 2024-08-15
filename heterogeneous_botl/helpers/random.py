#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def full_space_search_random(
    target_task_x: torch.Tensor,
    target_task_y: torch.Tensor,
    target_idxs: List[int],
    source_Xs: List[torch.Tensor],
    source_ys: List[torch.Tensor],
    source_idxs: List[List[int]],
    tkwargs: Dict[str, Any],
    search_space_bounds: Optional[List[int]] = None,
    full_feature_dim: Optional[int] = None,  # ignored
    fixed_imputation_constant: Optional[float] = None,  # ignored
) -> torch.Tensor:
    """
    Generate a random candidate in the search space.
    """
    search_space_bounds = search_space_bounds or [0.0, 1.0]
    scale = search_space_bounds[1] - search_space_bounds[0]
    candidate = (
        torch.rand(1, len(target_idxs), **tkwargs) * scale + search_space_bounds[0]
    )
    return candidate


def fixed_dataset_search_random(
    bo_acq_data_idxs: np.array,
    all_target_Xs: torch.Tensor,
    all_target_ys: torch.Tensor,
    target_idxs: List[int],
    source_Xs: List[torch.Tensor],
    source_ys: List[torch.Tensor],
    source_idxs: List[List[int]],
    tkwargs: Dict[str, Any],
    full_feature_dim: Optional[int] = None,  # ignored
    fixed_imputation_constant: Optional[float] = None,  # ignored
):
    """
    Get a unique sample from the fixed dataset using random sampling.
    """
    len_dataset = len(all_target_Xs)
    while True:
        idx = random.randint(0, len_dataset - 1)
        if idx not in bo_acq_data_idxs:
            bo_acq_data_idxs = np.concatenate((bo_acq_data_idxs, [idx]))
            return bo_acq_data_idxs
