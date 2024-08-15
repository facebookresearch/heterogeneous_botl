#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood

from heterogeneous_botl.models.model_utils import (
    get_covar_module_with_hvarfner_prior,
    get_gaussian_likelihood_with_log_normal_prior,
)


def get_fitted_mtgp_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
) -> MultiTaskGP:
    """
    Returns a fitted HeterogeneousMTGP model.
    """
    model = MultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
        output_tasks=[0],
        covar_module=get_covar_module_with_hvarfner_prior(
            ard_num_dims=train_X.shape[-1] - 1
        ),
        likelihood=get_gaussian_likelihood_with_log_normal_prior(),
        input_transform=Normalize(
            train_X.shape[-1], indices=list(range(train_X.shape[-1] - 1))
        ),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def find_common_indices(feature_indices_list: List[List[int]]):
    """
    This method finds common features (specified by indices)
    across all tasks. It also returns `common_indices_locations`
    which contains the location of the common features in each task.
    For example, if `feature_indices_list` is [[0, 1, 2], [1, 2, 3]],
    then the common features are [1, 2] and `common_indices_locations`
    are [[1, 2], [0, 1]].
    """
    common_indices = set(feature_indices_list[0])
    for indices in feature_indices_list[1:]:
        common_indices = common_indices.intersection(indices)
    common_indices = list(common_indices)
    common_indices_locations = []
    for indices in feature_indices_list:
        indices_locations = [i for i, x in enumerate(indices) if x in common_indices]
        common_indices_locations.append(indices_locations)
    return common_indices, common_indices_locations


def find_common_features_and_append_taskid(
    train_Xs: List[torch.Tensor],
    common_indices: List[int],
    common_indices_locations: List[List[int]],
    **tkwargs,
):
    """
    This method takes as input all the input tensors `train_Xs` and
    returns the tensors where the common features are extracted.
    """
    train_common_Xs = []
    for i in range(len(train_Xs)):
        X = torch.zeros(train_Xs[i].shape[0], len(common_indices) + 1, **tkwargs)
        X[:, :-1] = train_Xs[i][:, common_indices_locations[i]]
        X[:, -1] = i
        train_common_Xs.append(X)
    assert len(train_common_Xs) == len(train_Xs)
    return torch.cat(train_common_Xs)


def common_mtgp_full_space_search_bo(
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
    Perform one iteration of the BO loop w/
    MTGP imputed fitted over common features
    across all tasks and search over a
    fully specified space ([0, 1]^d).
    """
    search_space_bounds = search_space_bounds or [0.0, 1.0]
    common_indices, common_indices_locations = find_common_indices(
        feature_indices_list=[target_idxs, *source_idxs]
    )
    train_X = find_common_features_and_append_taskid(
        train_Xs=[target_task_x, *source_Xs],
        common_indices=common_indices,
        common_indices_locations=common_indices_locations,
        **tkwargs,
    )
    train_Y = torch.cat([target_task_y, *source_ys])

    model = get_fitted_mtgp_model(train_X=train_X, train_Y=train_Y)
    qei_acqf = qLogExpectedImprovement(model=model, best_f=target_task_y.max())

    def qei_acqf_aug(X):
        X_common = torch.zeros(*X.shape[:-1], len(common_indices) + 1, **tkwargs)
        X_common[..., :-1] = X[..., common_indices_locations[0]]
        X_common[..., -1] = 0
        return qei_acqf(X_common)

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


def common_mtgp_fixed_dataset_search_bo(
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
    Perform one iteration of the BO loop w/
    MTGP imputed fitted over common features
    across all tasks and searches over a fixed
    dataset given by (all_target_Xs, all_target_ys).
    `bo_acq_data_idxs` stores the list of indices
    picked by BO so far.
    """
    common_indices, common_indices_locations = find_common_indices(
        feature_indices_list=[target_idxs, *source_idxs]
    )
    target_task_x = all_target_Xs[bo_acq_data_idxs]
    target_task_y = all_target_ys[bo_acq_data_idxs]
    train_X = find_common_features_and_append_taskid(
        train_Xs=[target_task_x, *source_Xs],
        common_indices=common_indices,
        common_indices_locations=common_indices_locations,
        **tkwargs,
    )
    train_Y = torch.cat([target_task_y, *source_ys])

    model = get_fitted_mtgp_model(train_X=train_X, train_Y=train_Y)
    qei_acqf = qLogExpectedImprovement(model=model, best_f=target_task_y.max())

    def qei_acqf_aug(X):
        X_common = torch.zeros(*X.shape[:-1], len(common_indices) + 1, **tkwargs)
        X_common[..., :-1] = X[..., common_indices_locations[0]]
        X_common[..., -1] = 0
        return qei_acqf(X_common)

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
