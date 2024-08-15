#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import gpytorch
import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood

from heterogeneous_botl.models.imputed_heterogeneous_mtgp import (
    ImputedHeterogeneousMTGP,
)


def get_fitted_imputed_het_mtgp_model(
    target_task_x: torch.Tensor,
    target_task_y: torch.Tensor,
    target_idxs: List[int],
    source_Xs: List[torch.Tensor],
    source_ys: List[torch.Tensor],
    source_idxs: List[List[int]],
    full_feature_dim: int,
) -> ImputedHeterogeneousMTGP:
    """
    Returns a fitted HeterogeneousMTGP model.
    """
    with gpytorch.settings.debug(False):
        model = ImputedHeterogeneousMTGP(
            train_Xs=[target_task_x, *source_Xs],
            train_Ys=[target_task_y, *source_ys],
            train_Yvars=None,
            feature_indices=[target_idxs, *source_idxs],
            full_feature_dim=full_feature_dim,
            input_transform=Normalize(
                full_feature_dim + 1, indices=list(range(full_feature_dim))
            ),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    return model


def learned_imputed_mtgp_full_space_search_bo(
    target_task_x: torch.Tensor,
    target_task_y: torch.Tensor,
    target_idxs: List[int],
    source_Xs: List[torch.Tensor],
    source_ys: List[torch.Tensor],
    source_idxs: List[List[int]],
    full_feature_dim: int,
    tkwargs: Dict[str, Any],
    search_space_bounds: Optional[List[int]] = None,
    fixed_imputation_constant: Optional[float] = None,  # ignored
) -> torch.Tensor:
    """
    Perform one iteration of the BO loop w/
    HeterogeneousMTGP surrogate model over a
    fully specified search space ([0, 1]^d) as
    opposed to search over a fixed dataset which
    is below `het_mtgp_fixed_dataset_search_bo_loop`.
    """
    search_space_bounds = search_space_bounds or [0.0, 1.0]
    model = get_fitted_imputed_het_mtgp_model(
        target_task_x=target_task_x,
        target_task_y=target_task_y,
        target_idxs=target_idxs,
        source_Xs=source_Xs,
        source_ys=source_ys,
        source_idxs=source_idxs,
        full_feature_dim=full_feature_dim,
    )
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


def learned_imputed_mtgp_fixed_dataset_search_bo(
    bo_acq_data_idxs: np.array,
    all_target_Xs: torch.Tensor,
    all_target_ys: torch.Tensor,
    target_idxs: List[int],
    source_Xs: List[torch.Tensor],
    source_ys: List[torch.Tensor],
    source_idxs: List[List[int]],
    full_feature_dim: int,
    tkwargs: Dict[str, Any],
    fixed_imputation_constant: Optional[float] = None,  # ignored
) -> np.array:
    """
    Perform one iteration of the BO loop w/
    HeterogeneousMTGP surrogate model which
    searches over a fixed dataset given by
    (all_target_Xs, all_target_ys). `bo_acq_data_idxs`
    stores the list of indices picked by BO so far.
    """
    target_task_x = all_target_Xs[bo_acq_data_idxs]
    target_task_y = all_target_ys[bo_acq_data_idxs]

    model = get_fitted_imputed_het_mtgp_model(
        target_task_x=target_task_x,
        target_task_y=target_task_y,
        target_idxs=target_idxs,
        source_Xs=source_Xs,
        source_ys=source_ys,
        source_idxs=source_idxs,
        full_feature_dim=full_feature_dim,
    )
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
