#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from heterogeneous_botl.models.heterogeneous_mtgp import HeterogeneousMTGP
from heterogeneous_botl.models.model_utils import get_covar_module_with_hvarfner_prior


class ImputedHeterogeneousMTGP(HeterogeneousMTGP):
    """A multi-task GP model designed to operate on tasks from
    different search spaces. This model uses learned imputing features for
    each missing features of each task along with a base kernel.

    * The model is designed to work with a `MultiTaskDataset` that contains
        datasets with different features.
    * It uses a helper to embed the `X` coming from the sub-spaces into the
        full-feature space (+ task feature) before passing them down to the
        base `MultiTaskGP`.
    * The same helper is used in the `posterior` method to embed the `X` from
        the target task into the full dimensional space before evaluating the
        `posterior` method of the base class.
    * This model also reverts the `_split_inputs` overwrite from `HeterogenousMTGP`.
        Since we use the same base kernel as `MultiTaskGP`, we do not need
        the task feature added in `HeterogeneousMTGP`.
    """

    def __init__(
        self,
        train_Xs: List[Tensor],
        train_Ys: List[Tensor],
        train_Yvars: Optional[List[Tensor]],
        feature_indices: List[List[int]],
        full_feature_dim: int,
        rank: Optional[int] = None,
        all_tasks: Optional[List[int]] = None,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        """Construct a heterogeneous multi-task GP model from lists of inputs
        corresponding to each task.

        NOTE: This model assumes that the task 0 is the output / target task.
        It will only produce predictions for task 0.

        Args:
            train_Xs: A list of tensors of shape `(n_i x d_i)` where `d_i` is the
                dimensionality of the input features for task i.
                NOTE: These should not include the task feature!
            train_Ys: A list of tensors of shape `(n_i x 1)` containing the
                observations for the corresponding task.
            train_Yvars: An optional list of tensors of shape `(n_i x 1)` containing
                the observation variances for the corresponding task.
            feature_indices: A list of lists of integers specifying the indices
                mapping the features from a given task to the full tensor of features.
                The `i`th element of the list should contain `d_i` integers.
            full_feature_dim: The total number of features across all tasks. This
                does not include the task feature dimension.
            rank: The rank of the cross-task covariance matrix.
            all_tasks: By default, multi-task GPs infer the list of all tasks from
                the task features in `train_X`. This is an experimental feature that
                enables creation of multi-task GPs with tasks that don't appear in the
                training data. Note that when a task is not observed, the corresponding
                task covariance will heavily depend on random initialization and may
                behave unexpectedly.
            input_transform: An input transform that is applied in the model's
                forward pass. The transform should be compatible with the inputs
                from the full feature space with the task feature appended.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
        """
        self.train_Xs = [X.clone() for X in train_Xs]
        super().__init__(
            train_Xs=train_Xs,
            train_Ys=train_Ys,
            train_Yvars=train_Yvars,
            feature_indices=feature_indices,
            full_feature_dim=full_feature_dim,
            rank=rank,
            all_tasks=all_tasks,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        self.covar_module = get_covar_module_with_hvarfner_prior(
            ard_num_dims=self.num_non_task_features
        )
        # Initialize the imputing features as a tensor of num_tasks x full_feature_dim.
        # We will only use the missing values.
        # Each time imputing parameters are updated, we need to also
        # update the train_inputs of the GPyTorch model.
        self.register_parameter(
            "imputing_features",
            torch.nn.Parameter(
                torch.zeros(
                    len(train_Xs),
                    full_feature_dim,
                    dtype=train_Xs[0].dtype,
                    device=train_Xs[0].device,
                )
            ),
        )

    def _update_train_inputs(self) -> None:
        """Update the train inputs of the GPyTorch model using the current
        imputing features.
        """
        full_X = torch.cat(
            [
                self.map_to_full_tensor(X=X, task_index=i)
                for i, X in enumerate(self.train_Xs)
            ]
        )
        if self.training:
            full_X = self.transform_inputs(X=full_X)
        self.set_train_data(full_X, strict=False)

    def __call__(self, *args, **kwargs) -> MultivariateNormal:
        # Make sure we're using latest imputing features in train inputs when
        # evaluating the model.
        self._update_train_inputs()
        return super().__call__(*args, **kwargs)

    def map_to_full_tensor(self, X: Tensor, task_index: int) -> Tensor:
        """Map a tensor of task-specific features to the full tensor of features,
        utilizing the feature indices to map each feature to its corresponding
        position in the full tensor. Also append the task index as the last column.
        The missing columns of each task task will be filled with the learned
        imputing values while constructing the full feature tensor.

        Args:
            X: A tensor of shape `(n x d_i)` where `d_i` is the number of features
                in the original task dataset.
            task_index: The index of the task whose features are being mapped.

        Returns:
            A tensor of shape `(n x (self.full_feature_dim + 1))` containing the
            mapped features.

        Example:
            >>> # Suppose full feature dim is 3, the feature indices for
            >>> # task 5 are [2, 0], and the imputing values are 0.0 for all.
            >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> X_full = self.map_to_full_tensor(X=X, task_index=5)
            >>> # X_full = torch.tensor([[2.0, 0.0, 1.0, 5.0], [4.0, 0.0, 3.0, 5.0]])
        """
        X_full = torch.zeros(
            *X.shape[:-1], self.full_feature_dim + 1, dtype=X.dtype, device=X.device
        )
        if hasattr(self, "imputing_features"):  # This is needed to make __init__ work.
            X_full[..., :-1] = self.imputing_features[task_index]
        X_full[..., self.feature_indices[task_index]] = X
        X_full[..., -1] = task_index
        return X_full

    def _split_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Extracts base features and task indices from input data.

        Args:
            x: The full input tensor with trailing dimension of size `d + 1`.
                Should be of float/double data type.

        Returns:
            2-element tuple containing

            - A `q x d` or `b x q x d` (batch mode) tensor with trailing
            dimension made up of the `d` non-task-index columns of `x`, arranged
            in the order as specified by the indexer generated during model
            instantiation.
            - A `q` or `b x q` (batch mode) tensor of long data type containing
            the task indices.
        """
        return MultiTaskGP._split_inputs(self, x=x)
