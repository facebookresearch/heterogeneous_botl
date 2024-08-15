#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from heterogeneous_botl.models.heterogeneous_mtgp import HeterogeneousMTGP
from heterogeneous_botl.models.imputed_heterogeneous_mtgp import (
    ImputedHeterogeneousMTGP,
)
from heterogeneous_botl.models.multitask_conditional_kernel import (
    MultiTaskConditionalKernel,
)


class TestHeterogeneousMTGP(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.Xs = [torch.rand(5, 3), torch.rand(3, 2), torch.rand(2, 4)]
        self.Ys = [torch.rand(5, 1), torch.rand(3, 1), torch.rand(2, 1)]
        self.feature_indices = [[0, 1, 2], [0, 1], [0, 1, 3, 4]]
        self.full_feature_dim = 5

    def test_heterogeneous_mtgp(self) -> None:
        # Construct the model.
        model = HeterogeneousMTGP(
            train_Xs=self.Xs,
            train_Ys=self.Ys,
            train_Yvars=None,
            feature_indices=self.feature_indices,
            full_feature_dim=self.full_feature_dim,
        )
        self.assertEqual(model.train_inputs[0].shape, torch.Size([10, 6]))
        self.assertEqual(model._task_feature, 5)
        self.assertEqual(model._output_tasks, [0])
        self.assertEqual(model.num_tasks, 3)
        covar_module = model.covar_module
        self.assertIsInstance(covar_module, MultiTaskConditionalKernel)
        self.assertEqual(len(covar_module.kernels), 3)
        self.assertEqual(covar_module.binary_map, [[1, 1, 0], [1, 0, 0], [1, 0, 1]])

        # Evaluate the posterior.
        with self.assertRaisesRegex(UnsupportedError, "output_indices"):
            model.posterior(self.Xs[0], output_indices=[0, 1])
        posterior = model.posterior(self.Xs[0])
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior.distribution, MultivariateNormal)
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))
        posterior = model.posterior(self.Xs[0].repeat(3, 1, 1))
        self.assertEqual(posterior.mean.shape, torch.Size([3, 5, 1]))

    def test_identical_search_space(self) -> None:
        # Check that the model works fine with identical search spaces.
        # Construct the model.
        model = HeterogeneousMTGP(
            train_Xs=[torch.rand(5, 3), torch.rand(3, 3)],
            train_Ys=[torch.rand(5, 1), torch.rand(3, 1)],
            train_Yvars=None,
            feature_indices=[[0, 1, 2], [0, 1, 2]],
            full_feature_dim=3,
        )
        self.assertEqual(model.train_inputs[0].shape, torch.Size([8, 4]))
        covar_module = model.covar_module
        self.assertEqual(len(covar_module.kernels), 1)
        # Evaluate the posterior.
        posterior = model.posterior(self.Xs[0])
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))
        posterior = model.posterior(self.Xs[0].repeat(3, 1, 1))
        self.assertEqual(posterior.mean.shape, torch.Size([3, 5, 1]))

    def test_imputed_heterogeneous_mtgp(self) -> None:
        # Construct the model.
        model = ImputedHeterogeneousMTGP(
            train_Xs=self.Xs,
            train_Ys=self.Ys,
            train_Yvars=None,
            feature_indices=self.feature_indices,
            full_feature_dim=self.full_feature_dim,
        )
        self.assertEqual(model.train_inputs[0].shape, torch.Size([10, 6]))
        self.assertEqual(model._task_feature, 5)
        self.assertEqual(model._output_tasks, [0])
        self.assertEqual(model.num_tasks, 3)
        covar_module = model.covar_module
        self.assertIsInstance(covar_module, RBFKernel)
        self.assertEqual(model.imputing_features.shape, torch.Size([3, 5]))

        # Train the model.
        fit_gpytorch_mll(
            ExactMarginalLogLikelihood(model.likelihood, model),
            optimizer_kwargs={"options": {"maxiter": 1}},
            max_attempts=1,
        )
        self.assertGreaterEqual(
            # The parameters that are in the training data should not be updated.
            (model.imputing_features == 0.0).sum(),
            3 + 2 + 4,
        )

        # Evaluate the posterior.
        with self.assertRaisesRegex(UnsupportedError, "output_indices"):
            model.posterior(self.Xs[0], output_indices=[0, 1])
        posterior = model.posterior(self.Xs[0])
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior.distribution, MultivariateNormal)
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))
        posterior = model.posterior(self.Xs[0].repeat(3, 1, 1))
        self.assertEqual(posterior.mean.shape, torch.Size([3, 5, 1]))
