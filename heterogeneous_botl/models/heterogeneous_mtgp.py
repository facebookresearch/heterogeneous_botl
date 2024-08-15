#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from torch import Tensor

from heterogeneous_botl.models.model_utils import (
    get_gaussian_likelihood_with_log_normal_prior,
)
from heterogeneous_botl.models.multitask_conditional_kernel import (
    MultiTaskConditionalKernel,
)


class HeterogeneousMTGP(MultiTaskGP):
    """A multi-task GP model designed to operate on tasks from
    different search spaces. This model uses `MultiTaskConditionalKernel`.

    * The model is designed to work with a `MultiTaskDataset` that contains
        datasets with different features.
    * It uses a helper to embed the `X` coming from the sub-spaces into the
        full-feature space (+ task feature) before passing them down to the
        base `MultiTaskGP`.
    * The same helper is used in the `posterior` method to embed the `X` from
        the target task into the full dimensional space before evaluating the
        `posterior` method of the base class.
    * This model also overwrites the `_split_inputs` method. Instead of
        `x_basic`, we return the `X` with task feature included since this is
        used by the  `MultiTaskConditionalKernel` to identify the active
        dimensions of / the kernels to evaluate for the given input.
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
        use_scale_kernel: bool = False,
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
        self.full_feature_dim = full_feature_dim
        self.feature_indices = feature_indices
        full_X = torch.cat(
            [self.map_to_full_tensor(X=X, task_index=i) for i, X in enumerate(train_Xs)]
        )
        full_Y = torch.cat(train_Ys)
        full_Yvar = None if train_Yvars is None else torch.cat(train_Yvars)
        covar_module = MultiTaskConditionalKernel(
            feature_indices=feature_indices,
            use_scale_kernel=use_scale_kernel,
        )
        likelihood = (
            None  # Constructed in MultiTaskGP.
            if full_Yvar is not None
            else get_gaussian_likelihood_with_log_normal_prior()
        )
        super().__init__(
            train_X=full_X,
            train_Y=full_Y,
            task_feature=-1,
            train_Yvar=full_Yvar,
            covar_module=covar_module,
            likelihood=likelihood,
            output_tasks=[0],
            rank=rank,
            all_tasks=all_tasks,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

    def map_to_full_tensor(self, X: Tensor, task_index: int) -> Tensor:
        """Map a tensor of task-specific features to the full tensor of features,
        utilizing the feature indices to map each feature to its corresponding
        position in the full tensor. Also append the task index as the last column.
        The columns of the full tensor that are not used by the given task will be
        filled with zeros.

        Args:
            X: A tensor of shape `(n x d_i)` where `d_i` is the number of features
                in the original task dataset.
            task_index: The index of the task whose features are being mapped.

        Returns:
            A tensor of shape `(n x (self.full_feature_dim + 1))` containing the
            mapped features.

        Example:
            >>> # Suppose full feature dim is 3 and the feature indices for
            >>> # task 5 are [2, 0].
            >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> X_full = self.map_to_full_tensor(X=X, task_index=5)
            >>> # X_full = torch.tensor([[2.0, 0.0, 1.0, 5.0], [4.0, 0.0, 3.0, 5.0]])
        """
        X_full = torch.zeros(
            *X.shape[:-1], self.full_feature_dim + 1, dtype=X.dtype, device=X.device
        )
        X_full[..., self.feature_indices[task_index]] = X
        X_full[..., -1] = task_index
        return X_full

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> Union[GPyTorchPosterior, TransformedPosterior]:
        r"""Computes the posterior for the target task at the provided points.

        Args:
            X: A tensor of shape `batch_shape x q x d_0`, where `d_0` is the dimension
                of the feature space for task 0 (not including task indices) and `q` is
                the number of points considered jointly.
            output_indices: Not supported. Must be `None` or `[0]`. The model will
                only produce predictions for the target task regardless of
                the value of `output_indices`.
            observation_noise: If True, add observation noise from the respective
                likelihoods. If a Tensor, specifies the observation noise levels
                to add.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points.
        """
        if output_indices is not None and output_indices != [0]:
            raise UnsupportedError(
                "Heterogeneous MTGP does not support `output_indices`. "
            )
        X_full = self.map_to_full_tensor(X=X, task_index=0)
        return super().posterior(
            X=X_full,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def _split_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Returns x itself along with a tensor containing the task indices only.

        NOTE: This differs from the base class implementation because it returns
        the full tensor in place of `x_basic`. This is because the multi-task
        conditional kernel utilized the task feature for conditioning.

        Args:
            x: The full input tensor with trailing dimension of size
                `self.full_feature_dim + 1 + 1`.

        Returns:
            2-element tuple containing
            - The original tensor `x`.
            - A tensor of long data type containing the task indices.
        """
        task_idcs = x[..., self._task_feature : self._task_feature + 1].to(
            dtype=torch.long
        )
        return x, task_idcs
