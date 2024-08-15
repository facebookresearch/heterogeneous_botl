#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim.optimize import optimize_acqf

from heterogeneous_botl.helpers.common_mtgp_helpers import get_fitted_mtgp_model


def impute_missing_features_and_taskid(
    train_Xs: List[torch.Tensor],
    feature_indices: List[int],
    full_feature_dim: int,
    fixed_constant: float = 0.5,
    **tkwargs,
):
    """
    This method is used before for fitting the imputation based MTGP model.
    It takes in a list of input tensors `train_Xs` and imputes
    the missing features with a fixed constant value `fixed_constant`
    at locations specified by `feature_indices`.
    """
    assert len(train_Xs) == len(feature_indices)
    train_imputed_Xs = []
    for i in range(len(train_Xs)):
        missing_indices = set(range(full_feature_dim)) - set(feature_indices[i])
        missing_indices = list(missing_indices)
        X_imputed = torch.zeros(
            *train_Xs[i].shape[:-1], full_feature_dim + 1, **tkwargs
        )
        X_imputed[..., feature_indices[i]] = train_Xs[i]
        X_imputed[..., missing_indices] = fixed_constant
        X_imputed[..., -1] = i
        train_imputed_Xs.append(X_imputed)
    assert len(train_imputed_Xs) == len(train_Xs)
    return torch.cat(train_imputed_Xs)


def imputed_mtgp_full_space_search_bo(
    target_task_x: torch.Tensor,
    target_task_y: torch.Tensor,
    target_idxs: List[int],
    source_Xs: List[torch.Tensor],
    source_ys: List[torch.Tensor],
    source_idxs: List[List[int]],
    full_feature_dim: int,
    tkwargs: Dict[str, Any],
    fixed_imputation_constant: float,
    search_space_bounds: Optional[List[int]] = None,
    **ignore,
) -> torch.Tensor:
    """
    Perform one iteration of the BO loop w/
    MTGP imputed with fixed values over a
    fully specified search space ([0, 1]^d).
    """
    search_space_bounds = search_space_bounds or [0.0, 1.0]
    train_X = impute_missing_features_and_taskid(
        train_Xs=[target_task_x, *source_Xs],
        feature_indices=[target_idxs, *source_idxs],
        full_feature_dim=full_feature_dim,
        fixed_constant=fixed_imputation_constant,
        **tkwargs,
    )
    train_Y = torch.cat([target_task_y, *source_ys])
    model = get_fitted_mtgp_model(train_X=train_X, train_Y=train_Y)
    qei_acqf = qLogExpectedImprovement(model=model, best_f=target_task_y.max())

    def qei_acqf_aug(X):
        X = impute_missing_features_and_taskid(
            train_Xs=[X],
            feature_indices=[target_idxs],
            full_feature_dim=full_feature_dim,
            fixed_constant=fixed_imputation_constant,
            **tkwargs,
        )
        return qei_acqf(X)

    next_candidate, _ = optimize_acqf(
        acq_function=qei_acqf_aug,
        bounds=torch.tensor(
            [
                [search_space_bounds[0]] * len(target_idxs),
                [search_space_bounds[1]] * len(target_idxs),
            ],
            **tkwargs,
        ),
        q=1,
        num_restarts=20,
        raw_samples=1024,
    )
    return next_candidate


def imputed_mtgp_fixed_dataset_search_bo(
    bo_acq_data_idxs: np.array,
    all_target_Xs: torch.Tensor,
    all_target_ys: torch.Tensor,
    target_idxs: List[int],
    source_Xs: List[torch.Tensor],
    source_ys: List[torch.Tensor],
    source_idxs: List[List[int]],
    full_feature_dim: int,
    fixed_imputation_constant: float,
    tkwargs: Dict[str, Any],
) -> np.array:
    """
    Perform one iteration of the BO loop w/
    MTGP imputed with fixed values over a
    fixed dataset given by (all_target_Xs, all_target_ys).
    `bo_acq_data_idxs` stores the list of indices picked
    by BO so far.
    """
    target_task_x = all_target_Xs[bo_acq_data_idxs]
    target_task_y = all_target_ys[bo_acq_data_idxs]
    train_X = impute_missing_features_and_taskid(
        train_Xs=[target_task_x, *source_Xs],
        feature_indices=[target_idxs, *source_idxs],
        full_feature_dim=full_feature_dim,
        fixed_constant=fixed_imputation_constant,
        **tkwargs,
    )
    train_Y = torch.cat([target_task_y, *source_ys])

    model = get_fitted_mtgp_model(train_X=train_X, train_Y=train_Y)
    qei_acqf = qLogExpectedImprovement(model=model, best_f=target_task_y.max())

    def qei_acqf_aug(X):
        X = impute_missing_features_and_taskid(
            train_Xs=[X],
            feature_indices=[target_idxs],
            full_feature_dim=full_feature_dim,
            **tkwargs,
        )
        return qei_acqf(X)

    with torch.no_grad():
        acq_func_evals = torch.cat(
            [qei_acqf_aug(Xs) for Xs in all_target_Xs.unsqueeze(1).split(32)]
        )
    ids_sorted_by_acq = acq_func_evals.argsort(descending=True)

    for id_max_aquisition_all in ids_sorted_by_acq:
        if not id_max_aquisition_all.item() in bo_acq_data_idxs:
            id_max_aquisition = id_max_aquisition_all.item()
            break

    bo_acq_data_idxs = np.concatenate((bo_acq_data_idxs, [id_max_aquisition]))
    return bo_acq_data_idxs
