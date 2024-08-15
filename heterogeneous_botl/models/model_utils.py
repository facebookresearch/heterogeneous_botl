#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Union

import torch
from gpytorch import settings
from gpytorch.constraints import Interval
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior

EPS = 1e-8


class LogTransformedInterval(Interval):
    """Modification of the GPyTorch interval class.

    The Interval class in GPyTorch will map the parameter to the range [0, 1] before
    applying the inverse transform. We don't want to do this when using log as an
    inverse transform. This class will skip this step and apply the log transform
    directly to the parameter values so we can optimize log(parameter) under the bound
    constraints log(lower) <= log(parameter) <= log(upper).
    """

    def __init__(self, lower_bound, upper_bound, initial_value=None):
        r"""Initialize LogTransformedInterval."""
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=torch.exp,
            inv_transform=torch.log,
            initial_value=initial_value,
        )

        # Save the untransformed initial value
        self.register_buffer(
            "initial_value_untransformed",
            (
                torch.tensor(initial_value).to(self.lower_bound)
                if initial_value is not None
                else None
            ),
        )

        if settings.debug.on():
            max_bound = torch.max(self.upper_bound)
            min_bound = torch.min(self.lower_bound)
            if max_bound == math.inf or min_bound == -math.inf:
                raise RuntimeError(
                    "Cannot make an Interval directly with non-finite bounds. Use a "
                    "derived class like GreaterThan or LessThan instead."
                )

    def transform(self, tensor):
        if not self.enforced:
            return tensor

        transformed_tensor = self._transform(tensor)
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        if not self.enforced:
            return transformed_tensor

        tensor = self._inv_transform(transformed_tensor)
        return tensor


def get_gaussian_likelihood_with_log_normal_prior(
    batch_shape: Optional[torch.Size] = None,
) -> GaussianLikelihood:
    """Return Gaussian likelihood with a LogNormal(-4.0, 1.0) prior.
    This prior is based on Hvarfner et al (2024) "Vanilla Bayesian Optimization
    Performs Great in High Dimensions".
    https://arxiv.org/abs/2402.02229
    https://github.com/hvarfner/vanilla_bo_in_highdim

    Args:
        batch_shape: Batch shape for the likelihood.

    Returns:
        GaussianLikelihood with LogNormal(-4.0, 1.0) prior constrained to [1e-4, 1.0].
    """
    return GaussianLikelihood(
        noise_prior=LogNormalPrior(loc=-4.0, scale=1.0),
        noise_constraint=LogTransformedInterval(1e-4, 1, initial_value=1e-2),
        batch_shape=batch_shape or torch.Size(),
    )


def get_covar_module_with_hvarfner_prior(
    ard_num_dims: int,
    active_dims: Optional[List[int]] = None,
    batch_shape: Optional[torch.Size] = None,
    use_matern_kernel: bool = False,
    use_scale_kernel: bool = False,
    use_outputscale_prior: bool = False,
) -> Union[RBFKernel, MaternKernel, ScaleKernel]:
    """Returns an RBF or Matern kernel (with optional output scale) with priors
    from Hvarfner et al (2024) "Vanilla Bayesian Optimization Performs Great in
    High Dimensions".
    https://arxiv.org/abs/2402.02229
    https://github.com/hvarfner/vanilla_bo_in_highdim

    Args:
        ard_num_dims: Number of feature dimensions for the ARD kernel.
        active_dims: Active dims for the covar module. The kernel will be evaluated
            only using these columns of the input tensor.
        batch_shape: Batch shape for the covariance module.
        use_matern_kernel: Whether to use a Matern kernel. If False, uses an RBF kernel.
        use_scale_kernel: Whether to add an output scale using a ScaleKernel.
        use_outputscale_prior: Whether to add a Gamma(1.1, 0.2) prior
            on the outputscale.

    Returns:
        A Kernel constructed according to the given arguments.
    """
    base_class = MaternKernel if use_matern_kernel else RBFKernel
    base_kernel = base_class(
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
        lengthscale_prior=LogNormalPrior(
            loc=1.4 + math.log(ard_num_dims) * 0.5, scale=1.73205
        ),
        lengthscale_constraint=LogTransformedInterval(0.1, 100.00, initial_value=1.0),
    )
    if use_scale_kernel:
        return ScaleKernel(
            base_kernel=base_kernel,
            outputscale_prior=(
                GammaPrior(concentration=1.1, rate=0.2)
                if use_outputscale_prior
                else None
            ),
            outputscale_constraint=LogTransformedInterval(
                0.01, 100.0, initial_value=1.0
            ),
        )
    else:
        return base_kernel
