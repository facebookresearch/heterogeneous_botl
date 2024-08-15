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
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood

from heterogeneous_botl.helpers.common_mtgp_helpers import find_common_indices
from heterogeneous_botl.models.model_utils import (
    get_covar_module_with_hvarfner_prior,
    get_gaussian_likelihood_with_log_normal_prior,
)


def get_fitted_stgp_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
) -> SingleTaskGP:
    """
    Returns a fitted HeterogeneousMTGP model.
    """
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=get_covar_module_with_hvarfner_prior(
            ard_num_dims=train_X.shape[-1]
        ),
        likelihood=get_gaussian_likelihood_with_log_normal_prior(),
        input_transform=Normalize(train_X.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def stgp_full_space_search_bo(
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
    Perform one iteration of the BO loop w/ SingleTaskGP,
    and search over a fully specified space ([0, 1]^d).
    """
    search_space_bounds = search_space_bounds or [0.0, 1.0]
    model = get_fitted_stgp_model(train_X=target_task_x, train_Y=target_task_y)
    qei_acqf = qLogExpectedImprovement(model=model, best_f=target_task_y.max())

    next_candidate, _ = optimize_acqf(
        acq_function=qei_acqf,
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


def stgp_fixed_dataset_search_bo(
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
    Perform one iteration of the BO loop w/ SingleTaskGP,
    and searches over a fixed dataset given by (all_target_Xs, all_target_ys).
    `bo_acq_data_idxs` stores the list of indices picked by BO so far.
    """
    common_indices, common_indices_locations = find_common_indices(
        feature_indices_list=[target_idxs, *source_idxs]
    )
    target_task_x = all_target_Xs[bo_acq_data_idxs]
    target_task_y = all_target_ys[bo_acq_data_idxs]
    model = get_fitted_stgp_model(train_X=target_task_x, train_Y=target_task_y)
    qei_acqf = qLogExpectedImprovement(model=model, best_f=target_task_y.max())

    with torch.no_grad():
        acq_func_evals = torch.cat(
            [qei_acqf(Xs) for Xs in all_target_Xs.unsqueeze(1).split(32)]
        )
    ids_sorted_by_acq = acq_func_evals.argsort(descending=True)

    for id_max_aquisition_all in ids_sorted_by_acq:
        if not id_max_aquisition_all.item() in bo_acq_data_idxs:
            id_max_aquisition = id_max_aquisition_all.item()
            break

    bo_acq_data_idxs = np.concatenate((bo_acq_data_idxs, [id_max_aquisition]))
    return bo_acq_data_idxs
