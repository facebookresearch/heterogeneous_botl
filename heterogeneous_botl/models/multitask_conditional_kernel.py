#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
This is a variant of the conditional kernel that infers which sub-kernels to evaluate
from the task feature value. It also utilizes the active indices property of kernels
to avoid reordering of parameters.
"""

from typing import Dict, List, Set, Tuple

import torch
from gpytorch.kernels.kernel import Kernel
from torch import Tensor
from torch.nn import ModuleList

from heterogeneous_botl.models.model_utils import get_covar_module_with_hvarfner_prior


class DeltaKernel(Kernel):
    """This kernel evaluates `x1 == x2 == 1`."""

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        assert x1.shape[-1] == x2.shape[-1] == 1, "DeltaKernel expects 1D inputs!"
        x2_ = x2.transpose(-2, -1)
        return torch.where((x1 == x2_) & (x2_ == 1), 1.0, 0.0).to(x1)


class MultiTaskConditionalKernel(Kernel):
    r"""A base kernel for multi-task GPs with heterogeneous search
    spaces for tasks.

    * This kernel conditionally combines multiple sub-kernels to calculate covariances.
    * The kernel operates on `full_feature_dim + 1` dimensional inputs, with the `+ 1`
        dimension representing the task feature.
    * Given a list of indices representing the active feature dimensions for each task,
        the feature space is split into several non-overlapping subsets and a base
        kernel gets constructed for each of these subset dimensions.
    * The task feature is embedded into a binary tensor, which, together with a
        `DeltaKernel`, determines which of the sub-kernels are added together for
        the given inputs.
    * There is an additional kernel that operates over the binary
        embedding of task features.
    """

    def __init__(
        self,
        feature_indices: List[List[int]],
        task_feature_index: int = -1,
        use_matern_kernel: bool = False,
        use_scale_kernel: bool = False,
    ) -> None:
        r"""Initialize the kernel.

        Args:
            feature_indices: A list of lists of integers specifying the indices
                that select the features of a given task from the full tensor of
                features. The `i`th element of the list should contain `d_i`
                integers. These are the active indices for the given task.
            task_feature_index: Index of the task feature in the input tensor.
            use_matern_kernel: Whether to use a Matern kernel. If False, uses an
                RBF kernel.
            use_scale_kernel: Whether to add an output scale to each sub kernel.
        """
        super().__init__()
        self.use_matern_kernel = use_matern_kernel
        self.use_scale_kernel = use_scale_kernel
        self.task_feature_index: int = task_feature_index
        active_index_map, binary_map = map_subsets(
            subsets=find_subsets(feature_indices=feature_indices),
            feature_indices=feature_indices,
        )
        self.active_index_map: Dict[Tuple[int], List[int]] = active_index_map
        self.binary_map: List[List[int]] = binary_map
        self.kernels: ModuleList[Kernel] = self.construct_individual_kernels()
        self.binary_kernel: Kernel = get_covar_module_with_hvarfner_prior(
            ard_num_dims=len(self.kernels),
            use_matern_kernel=self.use_matern_kernel,
            use_scale_kernel=self.use_scale_kernel,
        )
        self.delta_kernel: DeltaKernel = DeltaKernel()

    def construct_individual_kernels(self) -> ModuleList:
        """Constructs the individual kernels corresponding to subsets
        of active feature dimensions.
        """
        kernels = ModuleList()
        for active_indices in self.active_index_map:
            kernels.append(
                get_covar_module_with_hvarfner_prior(
                    ard_num_dims=len(active_indices),
                    active_dims=active_indices,
                    use_matern_kernel=self.use_matern_kernel,
                    use_scale_kernel=self.use_scale_kernel,
                )
            )
        return kernels

    def map_task_to_binary(self, x_task: Tensor) -> Tensor:
        """Maps a tensor of task features to a binary tensor representing
        which kernels are active for the given task.

        Args:
            x_task: A tensor of task features of shape `batch x q`.

        Returns:
            A binary tensor of shape `batch x q x len(self.kernels)`.
            NOTE: The tensor has the same dtype as the input tensor.
            Returning a non-float tensor leads to downstream errors.
        """
        binary_map = torch.as_tensor(
            self.binary_map, dtype=x_task.dtype, device=x_task.device
        )
        return binary_map[x_task.long()]

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        r"""Evaluate the kernel on the given inputs.

        Args:
            x1: A `batch_shape x q1 x d`-dim tensor of inputs.
            x2: A `batch_shape x q2 x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x q1 x q2`-dim tensor of kernel values.
        """
        x1_binary = self.map_task_to_binary(x1[..., self.task_feature_index])
        x2_binary = self.map_task_to_binary(x2[..., self.task_feature_index])
        # This is a list of `batch_shape x q1 x q2`-dim tensors.
        kernel_evals = [k(x1, x2, **params) for k in self.kernels]
        # This is a `batch_shape x q1 x q2`-dim tensor.
        base_evals = self.binary_kernel(x1_binary, x2_binary, **params)
        # This is a list of `batch_shape x q1 x q2`-dim tensors.
        delta_evals = [
            self.delta_kernel(x1_b, x2_b, **params)
            for x1_b, x2_b in zip(
                x1_binary.split(1, dim=-1), x2_binary.split(1, dim=-1), strict=True
            )
        ]
        # Combine all kernels together to get the covariance.
        covar = base_evals
        for k, d in zip(kernel_evals, delta_evals, strict=True):
            covar = covar + k * d
        return covar


def find_subsets(feature_indices: List[List[int]]) -> List[Set[int]]:
    """Find the subsets of indices for which to construct sub-kernels.
    The goal is to find subsets of indices that are common across
    as many possible tasks.

    The main idea behind this implementation is to keep a running list
    of subsets. We will compare each index list with the elements of subsets,
    break them up and add to the list as needed.

    Args:
        feature_indices: A list of lists of integers specifying the indices
            mapping the features from a given task to the full tensor of features.

    Returns:
        A list of subsets of indices. All indices in the input must appear
        in exactly one of these subsets. When the subsets in the output
        are mapped to the inputs they are subsets of, each mapping should be
        unique. See the examples for more details.

    Examples:
        If input contains only one iterable, the output should be same
        as the input, cast to a set.
        If input contains two iterables, the output should be the intersection
        of the two inputs and the differences of the two. I.e., for an input of
        `[[1, 2, 3, 4], [1, 2, 5]]`, the output would be `[{1, 2}, {3, 4}, {5}]`.
        For larger inputs, the same logic applies. The key point is that we want
        the subsets to be as large as possible. For the above example,
        `[{1, 2}, {3}, {4}, {5}]` would not be acceptable since `{3}` and `{4}`
        can be joined together into a single subset of same inputs.
        However, if the inputs included a third iterable `[1, 2, 3]`, then
        `[{1, 2}, {3}, {4}, {5}]` would be the correct output since `3` appears
        in both inputs `[1, 2, 3, 4]` and `[1, 2, 3]`, but `4` only appears in
        the first one.
        The unit tests for this function provides some additional examples.
    """
    old_subsets = [set(feature_indices[0])]
    for idx_list in feature_indices[1:]:
        idx_set = set(idx_list)
        new_subsets = []
        for sub in old_subsets:
            # The idx_set contains a (possibly empty) subset of sub and potentially
            # other elements that are not in it. Break sub into two along the
            # intersection, remove common elements from idx_set and let
            # the loop continue.
            common = idx_set.intersection(sub)
            remaining = sub.difference(common)
            if common:
                new_subsets.append(common)
                idx_set = idx_set.difference(common)
            if remaining:
                new_subsets.append(remaining)
        # If there are elements in idx_set that were not in any of the subsets,
        # we check and add those as another subset here.
        if idx_set:
            new_subsets.append(idx_set)
        old_subsets = new_subsets
    return old_subsets


def map_subsets(
    subsets: List[Set[int]], feature_indices: List[List[int]]
) -> Tuple[Dict[Tuple[int], List[int]], List[List[int]]]:
    """Map the given list of subsets of indices to the indices of feature lists they
    are subsets of. Additionally, construct a reverse mapping, a list of length
    `len(feature_indices)`, where each element is a list of length `len(subsets)`.
    Reverse mapping can be thought of as a `len(feature_indices) x len(subsets)`
    matrix, where element (i, j) is 1 if `subsets[j]` is contained in
    `feature_indices[i]` and 0 otherwise.

    Args:
        subsets: A list of sets of indices. Obtained using `find_subsets`.
        feature_indices: A list of lists of integers specifying the indices
            mapping the features from a given task to the full tensor of features.

    Returns:
        A tuple of a dictionary and a list:
        A dictionary mapping each subset (cast to tuple) to the indices
            of feature lists it is subsets of.
        A list where each element of the list (with index i) contains a binary
            list (indexed by j) representing whether the corresponding subset
            (`subsets[j]`) is active for (i.e., a subset of) the corresponding
            feature list (`feature_indices[i]`).

    Examples:
        >>> feature_indices = [[1, 2, 3, 4], [1, 2, 5]]
        >>> subsets = find_subsets(feature_indices)
        >>> # `subsets` is `[{1, 2}, {3, 4}, {5}]`.
        >>> map_subsets(subsets, feature_indices)
        >>> # Should produce `{(1, 2): [0, 1], (3, 4): [0], (5): [1]}`
        >>> # and `[[1, 1, 0], [1, 0, 1]]`.
    """
    feature_index_map = {
        tuple(s): [
            i for i, idx_list in enumerate(feature_indices) if s.issubset(idx_list)
        ]
        for s in subsets
    }
    binary_map = [
        [int(s.issubset(idx_list)) for s in subsets] for idx_list in feature_indices
    ]
    return feature_index_map, binary_map
