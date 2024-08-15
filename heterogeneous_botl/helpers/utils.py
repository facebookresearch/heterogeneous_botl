#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from functools import partial
from typing import Callable, TypeVar

from heterogeneous_botl.helpers.common_mtgp_helpers import (
    common_mtgp_fixed_dataset_search_bo,
    common_mtgp_full_space_search_bo,
)
from heterogeneous_botl.helpers.fixed_imputed_mtgp_helpers import (
    imputed_mtgp_fixed_dataset_search_bo,
    imputed_mtgp_full_space_search_bo,
)
from heterogeneous_botl.helpers.het_mtgp_helpers import (
    het_mtgp_fixed_dataset_search_bo,
    het_mtgp_full_space_search_bo,
)
from heterogeneous_botl.helpers.learned_imputed_mtgp_helpers import (
    learned_imputed_mtgp_fixed_dataset_search_bo,
    learned_imputed_mtgp_full_space_search_bo,
)
from heterogeneous_botl.helpers.random import (
    fixed_dataset_search_random,
    full_space_search_random,
)
from heterogeneous_botl.helpers.stgp_helpers import (
    stgp_fixed_dataset_search_bo,
    stgp_full_space_search_bo,
)

T = TypeVar("T")


def get_method_from_name(method_name: str, full_space: bool) -> Callable:
    """Get the method by the given name.

    Args:
        method_name: Name of the method.
        full_space: Whether to use the full search space for optimization.
            If False, returns a variant that searches within given
            `all_target_Xs`.
    """
    if method_name == "random":
        return full_space_search_random if full_space else fixed_dataset_search_random
    if method_name == "single-task":
        return stgp_full_space_search_bo if full_space else stgp_fixed_dataset_search_bo
    if method_name == "het_mtgp":
        return (
            het_mtgp_full_space_search_bo
            if full_space
            else het_mtgp_fixed_dataset_search_bo
        )
    if method_name == "het_mtgp_scale":
        return partial(
            (
                het_mtgp_full_space_search_bo
                if full_space
                else het_mtgp_fixed_dataset_search_bo
            ),
            use_scale_kernel=True,
        )
    if method_name == "imputed_mtgp":
        return (
            imputed_mtgp_full_space_search_bo
            if full_space
            else imputed_mtgp_fixed_dataset_search_bo
        )
    if method_name == "common_mtgp":
        return (
            common_mtgp_full_space_search_bo
            if full_space
            else common_mtgp_fixed_dataset_search_bo
        )
    if method_name == "learned_imputed_mtgp":
        return (
            learned_imputed_mtgp_full_space_search_bo
            if full_space
            else learned_imputed_mtgp_fixed_dataset_search_bo
        )
    else:
        raise ValueError(f"Unknown method name {method_name}")


def with_retry_on_exception(*, func: Callable[..., T], **kwargs) -> T:
    """The method retries the given function up to 5 times
    if it fails with an exception.
    """
    num_failures = 0
    while True:
        try:
            return func(**kwargs)
        except Exception as e:  # pragma: no cover
            num_failures += 1
            if num_failures >= 5:
                raise e
            else:
                warnings.warn(
                    f"Failed to run {func}, {num_failures=}. Exception: {e}",
                    stacklevel=2,
                )
