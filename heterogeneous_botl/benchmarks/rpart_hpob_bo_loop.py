#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from logging import getLogger

import numpy as np
import torch

from heterogeneous_botl.helpers.utils import (
    get_method_from_name,
    with_retry_on_exception,
)
from heterogeneous_botl.hpo_b.hpob_handler import HPOBHandler
from heterogeneous_botl.hpo_b.hpob_utils import (
    get_test_data_by_search_id,
    get_train_data_by_search_id,
)

logger = getLogger(__name__)
MAX_DIM = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"dtype": dtype, "device": device}

# 'rpart': ['5636', '5859', '4796']
#        - [minsplit, minsplit.na, minbucket, cp, maxdepth, maxdepth.na]
#        - [minsplit, minsplit.na, minbucket, cp, maxdepth, maxdepth.na]
#        - [minsplit, minbucket, cp]


def setup_rpart_data(
    hpob_hdlr: HPOBHandler,
    n_source_samples: int,
    train_dataset_index: int,
    test_dataset_index: int,
):
    source_searchspace_ids = ["5636", "5859"]
    target_searchspace_id = "4796"
    source_Xs, source_ys = [], []
    for s in source_searchspace_ids:
        X, y = get_train_data_by_search_id(
            hpob_hdlr=hpob_hdlr,
            search_space_id=s,
            n_source_samples=n_source_samples,
            dataset_index=train_dataset_index,
        )
        X, y = torch.tensor(X, **tkwargs), torch.tensor(y, **tkwargs)
        random_indices = np.random.choice(
            range(X.shape[0]), n_source_samples, replace=False
        )
        source_Xs.append(X[random_indices])
        source_ys.append(y[random_indices])
    all_target_Xs, all_target_ys = get_test_data_by_search_id(
        hpob_hdlr=hpob_hdlr,
        search_space_id=target_searchspace_id,
        dataset_index=test_dataset_index,
    )
    all_target_Xs, all_target_ys = torch.tensor(all_target_Xs, **tkwargs), torch.tensor(
        all_target_ys, **tkwargs
    )
    source_tasks_ids = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    target_task_id = [0, 2, 3]
    return (
        source_Xs,
        source_ys,
        all_target_Xs,
        all_target_ys,
        source_tasks_ids,
        target_task_id,
    )


def rpart_main(
    method_name: str,
    seed: int,
    n_bo_steps: int = 35,
    n_source_samples: int = 30,
    n_init_target_samples: int = 5,
    train_dataset_index: int = 0,
    test_dataset_index: int = 0,
) -> None:
    logger.info(f"Running with method: {method_name} & seed {seed}.")
    method = get_method_from_name(method_name=method_name, full_space=False)

    hpob_hdlr = HPOBHandler()
    torch.manual_seed(seed)
    np.random.seed(seed)
    (
        source_Xs,
        source_ys,
        all_target_Xs,
        all_target_ys,
        source_idxs,
        target_idxs,
    ) = setup_rpart_data(
        hpob_hdlr,
        n_source_samples=n_source_samples,
        train_dataset_index=train_dataset_index,
        test_dataset_index=test_dataset_index,
    )
    config = {
        "source_idxs": source_idxs,
        "target_idxs": target_idxs,
        "n_bo_steps": n_bo_steps,
        "n_source_samples": n_source_samples,
        "n_init_target_samples": n_init_target_samples,
    }

    bo_acq_data_idxs = np.random.randint(
        low=0, high=len(all_target_ys), size=n_init_target_samples
    )
    logger.info(f"Starting observations: {all_target_ys[bo_acq_data_idxs].squeeze()}")
    running_optimum = torch.empty(n_bo_steps + n_init_target_samples, **tkwargs)
    for i in range(n_init_target_samples):
        running_optimum[i] = all_target_ys[bo_acq_data_idxs][: i + 1].max()
    for step in range(n_bo_steps):
        logger.info(f"step {step}")
        bo_acq_data_idxs = with_retry_on_exception(
            func=method,
            bo_acq_data_idxs=bo_acq_data_idxs,
            all_target_Xs=all_target_Xs,
            all_target_ys=all_target_ys,
            target_idxs=target_idxs,
            source_Xs=source_Xs,
            source_ys=source_ys,
            source_idxs=source_idxs,
            full_feature_dim=MAX_DIM,
            fixed_imputation_constant=0.5,
            tkwargs=tkwargs,
        )
        target_task_x = all_target_Xs[bo_acq_data_idxs]
        target_task_y = all_target_ys[bo_acq_data_idxs]
        running_idx = step + n_init_target_samples
        running_optimum[running_idx] = target_task_y.max()
        logger.info(f"bo_acq_data_idxs: {bo_acq_data_idxs}")
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
        "--train_dataset_index",
        type=int,
        default=0,
        help="Index of training (source) dataset"
        " from HPO-B [Possible choices: 0 to 50]",
    )
    parser.add_argument(
        "--test_dataset_index",
        type=int,
        default=0,
        help="Index of testing (target) dataset"
        " from HPO-B [Possible choices: {0, 1, 2, 3}]",
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
            f"{method_name}_rpart_{args.n_source_samples}_source"
            f"_{args.n_replications}_replications.pkl",
        )
        all_results = []
        for seed in range(len(args.n_replications)):
            res = rpart_main(
                method_name=method_name,
                seed=seed,
                n_bo_steps=args.n_bo_steps,
                n_source_samples=args.n_source_samples,
                n_init_target_samples=args.n_init_target_samples,
                train_dataset_index=args.train_dataset_index,
                test_dataset_index=args.test_dataset_index,
            )
            all_results.append(res)
            torch.save(output_path)
