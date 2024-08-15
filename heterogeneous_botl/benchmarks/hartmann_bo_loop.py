#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from logging import getLogger
from typing import List, Tuple

import torch
from botorch.test_functions.synthetic import Hartmann
from botorch.utils.sampling import draw_sobol_samples

from heterogeneous_botl.helpers.utils import (
    get_method_from_name,
    with_retry_on_exception,
)

logger = getLogger(__name__)
MAX_DIM = 6
hartmann_obj = Hartmann(negate=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"dtype": dtype, "device": device}


def compute_hartmann_objective(
    x_tensor: torch.Tensor, feature_idxs: List[int], fixed_constant: float
) -> torch.Tensor:
    fixed_aug_x = torch.ones(x_tensor.shape[0], MAX_DIM, **tkwargs) * fixed_constant
    fixed_aug_x[:, feature_idxs] = x_tensor
    objective_value = hartmann_obj(fixed_aug_x)
    return objective_value


def get_source_tasks_data(
    n_source_samples: int,
    source_idxs: List[List[int]],
    fixed_constant: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    source_Xs, source_ys = [], []
    for s_idx in source_idxs:
        source_task_x = draw_sobol_samples(
            bounds=torch.tensor([[0.0] * len(s_idx), [1.0] * len(s_idx)], **tkwargs),
            n=n_source_samples,
            q=1,
        ).squeeze(1)
        source_task_y = compute_hartmann_objective(
            x_tensor=source_task_x, feature_idxs=s_idx, fixed_constant=fixed_constant
        ).unsqueeze(1)
        source_Xs.append(source_task_x)
        source_ys.append(source_task_y)
    return source_Xs, source_ys


def get_target_task_data(
    n_target_samples: int, target_idxs: List[int], fixed_constant: float
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    target_task_x = draw_sobol_samples(
        bounds=torch.tensor(
            [[0.0] * len(target_idxs), [1.0] * len(target_idxs)], **tkwargs
        ),
        n=n_target_samples,
        q=1,
    ).squeeze(1)
    target_task_y = compute_hartmann_objective(
        x_tensor=target_task_x, feature_idxs=target_idxs, fixed_constant=fixed_constant
    ).unsqueeze(1)
    return target_task_x, target_task_y


def hartmann_main(
    method_name: str,
    source_idxs: List[List[int]],
    target_idxs: List[int],
    seed: int,
    n_bo_steps: int = 35,
    n_source_samples: int = 30,
    n_init_target_samples: int = 5,
    fixed_constant: float = 0.0,
    imputation_value: float = 0.5,
) -> None:
    logger.info(f"Running with method: {method_name} & seed {seed}.")
    method = get_method_from_name(method_name=method_name, full_space=True)
    config = {
        "source_idxs": source_idxs,
        "target_idxs": target_idxs,
        "n_bo_steps": n_bo_steps,
        "n_source_samples": n_source_samples,
        "n_init_target_samples": n_init_target_samples,
        "fixed_constant": fixed_constant,
    }
    torch.manual_seed(seed)
    source_Xs, source_ys = get_source_tasks_data(
        n_source_samples=n_source_samples,
        source_idxs=source_idxs,
        fixed_constant=fixed_constant,
    )
    target_task_x, target_task_y = get_target_task_data(
        n_target_samples=n_init_target_samples,
        target_idxs=target_idxs,
        fixed_constant=fixed_constant,
    )
    logger.info(f"best value in the initial data: {target_task_y.max()}")
    running_optimum = torch.empty(n_bo_steps + n_init_target_samples, **tkwargs)
    for i in range(n_init_target_samples):
        running_optimum[i] = target_task_y[: i + 1].max()
    for step in range(n_bo_steps):
        logger.info(f"step {step}")
        next_candidate = with_retry_on_exception(
            func=method,
            target_task_x=target_task_x,
            target_task_y=target_task_y,
            target_idxs=target_idxs,
            source_Xs=source_Xs,
            source_ys=source_ys,
            source_idxs=source_idxs,
            full_feature_dim=MAX_DIM,
            fixed_imputation_constant=imputation_value,
            tkwargs=tkwargs,
        )
        next_candidate_obj_val = compute_hartmann_objective(
            x_tensor=next_candidate,
            feature_idxs=target_idxs,
            fixed_constant=fixed_constant,
        ).unsqueeze(1)
        target_task_x = torch.cat((target_task_x, next_candidate), dim=0)
        target_task_y = torch.cat((target_task_y, next_candidate_obj_val), dim=0)
        running_idx = step + n_init_target_samples
        running_optimum[running_idx] = target_task_y.max()
        logger.info(f"best value at iteration {step}: {running_optimum[running_idx]}")
    bo_results = {
        "target_task_x": target_task_x,
        "target_task_y": target_task_y,
        "config": config,
        "source_Xs": source_Xs,
        "source_ys": source_ys,
        "running_optimum": running_optimum,
    }
    return bo_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--fixed_constant",
        type=float,
        default=0.0,
        help="Fixed value to use for unobserved features to evaluate the objective.",
    )
    parser.add_argument(
        "--imputation_value",
        type=float,
        default=0.5,
        help="Fixed imputation value.",
    )
    parser.add_argument(
        "--n_bo_steps",
        type=int,
        default=35,
        help="No. of BO steps in each replication",
    )
    parser.add_argument(
        "--n_source_samples",
        type=int,
        default=30,
        help="Size of source task dataset",
    )
    parser.add_argument(
        "--n_init_target_samples",
        type=int,
        default=5,
        help="Size of initial target task dataset",
    )
    parser.add_argument(
        "--n_replications",
        type=int,
        default=10,
        help="No. of replications",
    )

    args = parser.parse_args()
    source_idxs = [[0, 1, 2, 3]]
    target_idxs = [0, 1, 2, 3, 4, 5]
    root_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    results_dir = os.path.join(root_path, "results")
    for method_name in [
        "random",
        "single-task",
        "het_mtgp",
        "imputed_mtgp",
        "common_mtgp",
        "learned_imputed_mtgp",
    ]:
        output_path = os.path.join(
            results_dir,
            f"{method_name}_hartmann_{args.n_source_samples}_source"
            f"_{args.n_replications}_replications.pkl",
        )
        all_results = []
        for seed in len(args.n_replications):
            res = hartmann_main(
                method_name=method_name,
                source_idxs=source_idxs,
                target_idxs=target_idxs,
                seed=seed,
                n_bo_steps=args.n_bo_steps,
                n_source_samples=args.n_source_samples,
                n_init_target_samples=args.n_init_target_samples,
                fixed_constant=args.fixed_constant,
                imputation_value=args.imputation_value,
            )
            all_results.append(res)
            torch.save(output_path)
