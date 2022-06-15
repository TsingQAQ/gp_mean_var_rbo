# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module is the home of the sampling functionality required by Trieste's
acquisition functions.
"""

from __future__ import annotations

from abc import ABC
from typing import Callable, Optional

import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import bisect
from scipy.stats.qmc import MultivariateNormalQMC

from ..data import Dataset
from ..models import ProbabilisticModel
from ..models.gpflux.future.layers.basis_functions.fourier_features.quadrature import (
    QuadratureFourierFeatures,
)
from ..models.gpflux.future.layers.basis_functions.fourier_features.random import (
    RandomFourierFeaturesCosine as RandomFourierFeatures,
)
from ..types import TensorType
from ..utils import DEFAULTS
from ..utils.wrapper import sequential_batch


class ThompsonSampler(ABC):
    r"""
    A :class:`ThompsonSampler` samples either the minimum values or minimisers of a function
    modeled by an underlying :class:`ProbabilisticModel` across a  discrete set of points.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel, sample_min_value: bool = False):
        """
        :param sample_size: The desired number of samples.
        :param model: The model to sample from.
        :sample_min_value: If True then sample from the minimum value of the function,
            else sample the function's minimiser.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        self._sample_min_value = sample_min_value
        self._sample_size = sample_size
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}(
        {self._sample_size!r},
        {self._model!r},
        {self._sample_min_value})
        """


class ExactThompsonSampler(ThompsonSampler):
    r"""
    This sampler provides exact Thompson samples of the objective function's
    minimiser :math:`x^*` over a discrete set of input locations.

    Although exact Thompson sampling is costly (incuring with an :math:`O(N^3)` complexity to
    sample over a set of `N` locations), this method can be used for any probabilistic model
    with a sampling method.

    """

    def sample(self, model: ProbabilisticModel, sample_size: int, at: TensorType) -> TensorType:
        """
        Return exact samples from either the objective function's minimser or its minimal value
        over the candidate set `at`.

        :param model: The model to sample from.
        :param sample_size: The desired number of samples.
        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[S, D]` (where `S` is the `sample_size`) if sampling
            the function's minimser or shape `[S, 1]` if sampling the function's mimimal value.
        :raise ValueError: If ``at`` has an invalid shape or if ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        samples = model.sample(at, sample_size)  # [S, N, 1]

        if self._sample_min_value:
            thompson_samples = tf.reduce_min(samples, axis=1)  # [S, 1]
        else:
            samples_2d = tf.squeeze(samples, -1)  # [S, N]
            indices = tf.math.argmin(samples_2d, axis=1)
            thompson_samples = tf.gather(at, indices)  # [S, D]

        return thompson_samples


class GumbelSampler(ThompsonSampler):
    r"""
    This sampler follows :cite:`wang2017max` and yields approximate samples of the objective
    minimum value :math:`y^*` via the empirical cdf :math:`\operatorname{Pr}(y^*<y)`. The cdf
    is approximated by a Gumbel distribution

    .. math:: \mathcal G(y; a, b) = 1 - e^{-e^\frac{y - a}{b}}

    where :math:`a, b \in \mathbb R` are chosen such that the quartiles of the Gumbel and cdf match.
    Samples are obtained via the Gumbel distribution by sampling :math:`r` uniformly from
    :math:`[0, 1]` and applying the inverse probability integral transform
    :math:`y = \mathcal G^{-1}(r; a, b)`.

    Note that the :class:`GumbelSampler` can only sample a function's minimal value and not
    its minimiser.
    """

    def __init__(self, sample_min_value: bool = False):
        """
        :sample_min_value: If True then sample from the minimum value of the function,
            else sample the function's minimiser.
        """

        if not sample_min_value:
            raise ValueError(
                f"""
                Gumbel samplers can only sample a function's minimal value,
                however received sample_min_value={sample_min_value}
                """
            )

        super().__init__(sample_min_value)

    def sample(self, model: ProbabilisticModel, sample_size: int, at: TensorType) -> TensorType:
        """
        Return approximate samples from of the objective function's minimum value.

        :param model: The model to sample from.
        :param sample_size: The desired number of samples.
        :param at: Points at where to fit the Gumbel distribution, with shape `[N, D]`, for points
            of dimension `D`. We recommend scaling `N` with search space dimension.
        :return: The samples, of shape `[S, 1]`, where `S` is the `sample_size`.
        :raise ValueError: If ``at`` has an invalid shape or if ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        try:
            fmean, fvar = model.predict_y(at)
        except NotImplementedError:
            fmean, fvar = model.predict(at)

        fsd = tf.math.sqrt(fvar)

        def probf(y: tf.Tensor) -> tf.Tensor:  # Build empirical CDF for Pr(y*^hat<y)
            unit_normal = tfp.distributions.Normal(tf.cast(0, fmean.dtype), tf.cast(1, fmean.dtype))
            log_cdf = unit_normal.log_cdf(-(y - fmean) / fsd)
            return 1 - tf.exp(tf.reduce_sum(log_cdf, axis=0))

        left = tf.reduce_min(fmean - 5 * fsd)
        right = tf.reduce_max(fmean + 5 * fsd)

        def binary_search(val: float) -> float:  # Find empirical interquartile range
            return bisect(lambda y: probf(y) - val, left, right, maxiter=10000)

        q1, q2 = map(binary_search, [0.25, 0.75])

        log = tf.math.log
        l1 = log(log(4.0 / 3.0))
        l2 = log(log(4.0))
        b = (q1 - q2) / (l1 - l2)
        a = (q2 * l1 - q1 * l2) / (l1 - l2)

        uniform_samples = tf.random.uniform([sample_size], dtype=fmean.dtype)
        gumbel_samples = log(-log(1 - uniform_samples)) * tf.cast(b, fmean.dtype) + tf.cast(
            a, fmean.dtype
        )
        gumbel_samples = tf.expand_dims(gumbel_samples, axis=-1)  # [S, 1]
        return gumbel_samples


class ThompsonSamplerFromTrajectory(ThompsonSampler):
    r"""
    This sampler provides approximate Thompson samples of the objective function's
    minimiser :math:`x^*` by minimizing approximate trajectories sampled from the
    underlying probabilistic model. This sampling method can be used for any
    probabilistic model with a :meth:`trajectory_sampler` method.
    """

    def sample(self, model: ProbabilisticModel, sample_size: int, at: TensorType) -> TensorType:
        """
        Return approximate samples from either the objective function's minimser or its minimal
        value over the candidate set `at`.

        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        super().__init__(sample_size, model)

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([0, 0, sample_size], dtype=tf.float64), shape=[None, None, sample_size]
        )  # [0, 0, S]

        # for some reason graph compilation is resulting in self._eps reporting the wrong shape
        # we therefore use an extra boolean variable to keep track of whether it's initialised
        self._initialized = tf.Variable(False)

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`BatchReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`BatchReparametrizationSampler` instances will produce different samples.

        :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            :class:`BatchReparametrizationSampler`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The samples, of shape `[..., S, B, L]`, where `S` is the `sample_size`, `B` the
            number of points per batch, and `L` the dimension of the model's predictive
            distribution.
        :raise ValueError (or InvalidArgumentError): If any of the following are true:
            - ``at`` is a scalar.
            - The batch size `B` of ``at`` is not positive.
            - The batch size `B` of ``at`` differs from that of previous calls.
            - ``jitter`` is negative.
        """
        tf.debugging.assert_rank_at_least(at, 2)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        batch_size = at.shape[-2]

        tf.debugging.assert_positive(batch_size)

        if self._initialized:
            tf.debugging.assert_equal(
                batch_size,
                tf.shape(self._eps)[-2],
                f"{type(self).__name__} requires a fixed batch size. Got batch size {batch_size}"
                f" but previous batch size was {tf.shape(self._eps)[-2]}.",
            )
        mean, cov = self._model.predict_joint(at)  # [..., B, L], [..., L, B, B]

        if not self._initialized:
            self._eps.assign(
                tf.random.normal(
                    [tf.shape(mean)[-1], batch_size, self._sample_size], dtype=tf.float64
                )  # [L, B, S]
            )
            self._initialized.assign(True)

        identity = tf.eye(batch_size, dtype=cov.dtype)  # [B, B]
        cov_cholesky = tf.linalg.cholesky(cov + jitter * identity)  # [..., L, B, B]

        variance_contribution = cov_cholesky @ tf.cast(self._eps, cov.dtype)  # [..., L, B, S]

        leading_indices = tf.range(tf.rank(variance_contribution) - 3)
        absolute_trailing_indices = [-1, -2, -3] + tf.rank(variance_contribution)
        new_order = tf.concat([leading_indices, absolute_trailing_indices], axis=0)

        return mean[..., None, :, :] + tf.transpose(variance_contribution, new_order)


TrajectoryFunction = Callable[[TensorType], TensorType]
"""
Type alias for trajectory functions.

An :const:`TrajectoryFunction` evaluates a particular sample at a set of `N` query
points (each of dimension `D`) i.e. takes input of shape `[N, D]` and returns
shape `[N, 1]`.

A key property of these trajectory functions is that the same sample draw is evaluated
for all queries. This property is known as consistency.
"""


class helper_lazy_theta_posterior:
    """
    a helper lay theta posterior, its usage is if no 'sample' called (i.e., just get posterior mean),
    we don't do Cholesky decomposition beforehand
    """

    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov
        self._sample_method = None
        self._distribution = None

    @property
    def loc(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    def sample(self, sample_num, method: str = "qMC"):
        if method == "standard":
            if self._distribution is None or self._sample_method != method:
                self._sample_method = method
                _theta_posterior_chol_covariance = tf.linalg.cholesky(self.cov)

                self._distribution = tfp.distributions.MultivariateNormalTriL(
                    self._mean, _theta_posterior_chol_covariance
                )
            return self._distribution.sample(sample_num)
        else:
            assert method == "qMC", NotImplementedError
            if self._distribution is None or self._sample_method != method:
                self._sample_method = method
                _theta_posterior_chol_covariance = tf.linalg.cholesky(self.cov)
                self._distribution = MultivariateNormalQMC(
                    self._mean, cov_root=tf.transpose(_theta_posterior_chol_covariance)
                )
            return self._distribution.random(sample_num)


class RandomFourierFeatureThompsonSampler(ThompsonSampler):
    r"""
    This class builds functions that approximate a trajectory sampled from an underlying Gaussian
    process model. For tractibility, the Gaussian process is approximated with a Bayesian
    Linear model across a set of features sampled from the Fourier feature decomposition of
    the model's kernel. See :cite:`hernandez2014predictive` for details.

    Achieving consistency (ensuring that the same sample draw for all evalutions of a particular
    trajectory function) for exact sample draws from a GP is prohibitively costly because it scales
    cubically with the number of query points. However, finite feature representations can be
    evaluated with constant cost regardless of the required number of queries.

    In particular, we approximate the Gaussian processes' posterior samples as the finite feature
    approximation

    .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i

    where :math:`\phi_i` are m Fourier features and :math:`\theta_i` are
    feature weights sampled from a posterior distribution that depends on the feature values at the
    model's datapoints.

    Our implementation follows :cite:`hernandez2014predictive`, with our calculations
    differing slightly depending on properties of the problem. In particular,  we used different
    calculation strategies depending on the number of considered features m and the number
    of data points n.

    If :math:`m<n` then we follow Appendix A of :cite:`hernandez2014predictive` and calculate the
    posterior distribution for :math:`\theta` following their Bayesian linear regression motivation,
    i.e. the computation revolves around an O(m^3)  inversion of a design matrix.

    If :math:`n<m` then we use the kernel trick to recast computation to revolve around an O(n^3)
    inversion of a gram matrix. As well as being more efficient in early BO
    steps (where :math:`n<m`), this second computation method allows must larger choices
    of m (as required to approximate very flexible kernels).
    """

    def __init__(
        self,
        sample_size: int,
        model: ProbabilisticModel,
        dataset: Dataset,
        sample_min_value: bool = False,
        num_features: int = 1000,
        fourier_feature_method="rff",
    ):
        """
        :param sample_size: The desired number of samples.
        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[S, D]` (where `S` is the `sample_size`) if sampling
            the function's minimser or shape `[S, 1]` if sampling the function's mimimal value.
        :raise ValueError: If ``at`` has an invalid shape or if ``sample_size`` is not positive.
        """

        super().__init__(sample_size, model, sample_min_value)

        if len(dataset.query_points) == 0:
            raise ValueError("Dataset must be populated.")

        self._dataset = dataset
        self._model = model

        tf.debugging.assert_positive(num_features)
        self._num_features = num_features  # m
        self._num_data = len(self._dataset.query_points)  # n
        self._fourier_feature_method = fourier_feature_method
        self._theta_posterior_trig = None

        try:
            self._noise_variance = model.get_observation_noise()
            self._kernel = model.get_kernel()
        except (NotImplementedError, AttributeError):
            raise ValueError(
                """
            Thompson sampling with random Fourier features only currently supports models
            with a Gaussian likelihood and an accessible kernel attribute.
            """
            )
        if fourier_feature_method == "rff":
            self._feature_functions = RandomFourierFeatures(
                self._kernel, self._num_features, dtype=self._dataset.query_points.dtype
            )  # prep feature functions at data
        elif fourier_feature_method == "qff":
            self._feature_functions = QuadratureFourierFeatures(
                self._kernel, self._num_features, dtype=self._dataset.query_points.dtype
            )
        else:
            raise NotImplementedError(
                f"fourier_feature_method: {fourier_feature_method} not understood"
            )

        if (
            self._num_features < self._num_data
        ):  # if m < n  then calculate posterior in design space (an m*m matrix inversion)
            self._theta_posterior = self._prepare_theta_posterior_in_design_space()
        else:  # if n <= m  then calculate posterior in gram space (an n*n matrix inversion)
            self._theta_posterior = self._prepare_theta_posterior_in_gram_space()

        self._pre_calc = False  # Flag so we only calculate the posterior for the weights once.

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}(
        {self._sample_size!r},
        {self._model!r},
        {self._sample_min_value!r},
        {self._num_features!r})
        """

    def _prepare_theta_posterior_in_design_space(self) -> tfp.distributions.Distribution:
        # Calculate the posterior of theta (the feature weights) in the design space. This
        # distribution is a Gaussian
        #
        # .. math:: \theta \sim N(D^{-1}\Phi^Ty,D^{-1}\sigma^2)
        #
        # where the [m,m] design matrix :math:`D=(\Phi^T\Phi + \sigma^2I_m)` is defined for
        # the [n,m] matrix of feature evaluations across the training data :math:`\Phi`
        # and observation noise variance :math:`\sigma^2`.

        # [n, m] for rff, [n, 2m] for qff: first m are cos, then sin
        phi = self._feature_functions(self._dataset.query_points)
        D = tf.matmul(phi, phi, transpose_a=True)  # [m, m] or [2m, 2m]
        # s = self._noise_variance * tf.eye(self._num_features, dtype=phi.dtype)
        s = self._noise_variance * tf.eye(D.shape[0], dtype=phi.dtype)
        L = tf.linalg.cholesky(D + s)
        # D_inv = tf.linalg.cholesky_solve(L, tf.eye(self._num_features, dtype=phi.dtype))
        D_inv = tf.linalg.cholesky_solve(L, tf.eye(D.shape[0], dtype=phi.dtype))

        theta_posterior_mean = tf.matmul(
            D_inv, tf.matmul(phi, self._dataset.observations, transpose_a=True)
        )[
            :, 0
        ]  # [m,] for rff, [2m, ] for qff

        # [m, m] for rff, [2m, 2m] for qff
        return helper_lazy_theta_posterior(theta_posterior_mean, D_inv * self._noise_variance)

    def _prepare_theta_posterior_in_gram_space(self) -> tfp.distributions.Distribution:
        # Calculate the posterior of theta (the feature weights) in the gram space.
        #
        #  .. math:: \theta \sim N(\Phi^TG^{-1}y,I_m - \Phi^TG^{-1}\Phi)
        #
        # where the [n,n] gram matrix :math:`G=(\Phi\Phi^T + \sigma^2I_n)` is defined for the [n,m]
        # matrix of feature evaluations across the training data :math:`\Phi` and
        # observation noise variance :math:`\sigma^2`.
        # assert self._fourier_feature_method == 'rff'
        # [n, m] for rff, [N, 2 * (m^d)] for qff for qff: first m are cos, then sin
        phi = self._feature_functions(self._dataset.query_points)
        G = tf.matmul(phi, phi, transpose_b=True)  # [n, n]
        s = self._noise_variance * tf.eye(self._num_data, dtype=phi.dtype)
        L = tf.linalg.cholesky(G + s)
        L_inv_phi = tf.linalg.triangular_solve(L, phi)  # [n, m]
        L_inv_y = tf.linalg.triangular_solve(L, self._dataset.observations)  # [n, 1]

        theta_posterior_mean = tf.tensordot(tf.transpose(L_inv_phi), L_inv_y, [[-1], [-2]])[
            :, 0
        ]  # [m,]
        if self._fourier_feature_method == "rff":
            theta_posterior_covariance = tf.eye(self._num_features, dtype=phi.dtype) - tf.tensordot(
                tf.transpose(L_inv_phi), L_inv_phi, [[-1], [-2]]
            )  # [m, m]
        else:  # 'qff'
            input_dim = self._dataset.query_points.shape[-1]
            theta_posterior_covariance = tf.eye(
                2 * (self._num_features ** input_dim), dtype=phi.dtype
            ) - tf.tensordot(
                tf.transpose(L_inv_phi), L_inv_phi, [[-1], [-2]]
            )  # [m, m]
        return helper_lazy_theta_posterior(theta_posterior_mean, theta_posterior_covariance)

    def get_trajectory(
        self,
        sample_size=1,
        return_sample: bool = False,
        get_mean: bool = False,
        sample_method: str = "qMC",
    ) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by sampling weights
        and evaluating the feature functions.

        :return: A trajectory function representing an approximate trajectory from the Gaussian
            process, taking an input of shape `[N, D]` and returning shape `[N, 1]`
        """
        if get_mean is False:
            theta_sample = self._theta_posterior.sample(
                sample_size, sample_method
            )  # [sample_size, m] for rff; [sample_size, 2m] for qff
        else:
            theta_sample = self._theta_posterior.loc[tf.newaxis]  # [1, 2m]

        def trajectory(x: TensorType) -> TensorType:
            """
            :param x [N, dim]
            :return [sample_size, N, 1]
            """
            feature_evaluations = self._feature_functions(x)  # [N, m] for rff; [1, 2m] for qff
            # return tf.matmul(feature_evaluations, theta_sample, transpose_b=True)  # [N,1]
            # [sample_size, m/2m] * [m/2m, N] -> [sample_size, N]
            return tf.matmul(theta_sample, feature_evaluations, transpose_b=True)[
                ..., tf.newaxis
            ]  # [N,1]

        return trajectory if not return_sample else (trajectory, theta_sample)

    def sample(self, at: TensorType) -> TensorType:
        """
        Return approximate samples from either the objective function's minimser or its minimal
        value over the candidate set `at`.
        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[S, D]` (where `S` is the `sample_size`) if sampling
            the function's minimser or shape `[S, 1]` if sampling the function's mimimal value.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape.
        """
        tf.debugging.assert_shapes([(at, ["N", None])])

        if self._sample_min_value:
            thompson_samples = tf.zeros([0, 1], dtype=at.dtype)  # [0,1]
        else:
            thompson_samples = tf.zeros([0, tf.shape(at)[1]], dtype=at.dtype)  # [0,D]

        for _ in tf.range(self._sample_size):
            sampled_trajectory = self.get_trajectory()
            evaluated_trajectory = sampled_trajectory(at)  # [N, 1]
            if self._sample_min_value:
                sample = tf.reduce_min(evaluated_trajectory, keepdims=True)  # [1, 1]
            else:
                sample = tf.gather(at, tf.math.argmin(evaluated_trajectory))  # [1, D]

            thompson_samples = tf.concat([thompson_samples, sample], axis=0)

        return thompson_samples  # [S, D] or [S, 1]


class VarianceSampler(ABC):
    def feature_function_propagate_mean(self, inputs):
        pass

    def feature_function_propagate_var(self, inputs):
        pass

    def get_mean_trajectory(self) -> TrajectoryFunction:
        pass

    def get_var_trajectory(self) -> TrajectoryFunction:
        pass


class RFFVarianceSampler(RandomFourierFeatureThompsonSampler, VarianceSampler):
    """
    Sample and calculate the variance distribution
    """

    def __init__(
        self,
        noise_type: str,
        model: ProbabilisticModel,
        dataset: Dataset,
        sample_min_value: bool = False,
        num_features: int = 1000,
        **kwargs,
    ):
        assert noise_type in ["normal", "uniform"]
        if noise_type == "normal":
            # scale the normal properly
            self.noise_type = noise_type
            self.noise_cov = kwargs["noise"]
        else:
            assert noise_type == "uniform"
            self.noise_type = noise_type
            self.noise = kwargs["noise"]
            tf.debugging.assert_shapes([(self.noise, ("D"))])
        super().__init__(
            1,
            model,
            dataset,
            sample_min_value=sample_min_value,
            num_features=num_features,
        )

    def feature_function_propagate_mean(self, inputs):
        """
        calculate feature function propagated by mean
        """
        if self.noise_type == "normal":
            # c remain unchanged
            c = tf.sqrt(
                2 * self._feature_functions.kernel.variance / self._feature_functions.n_components
            )
            # input: scaled_inputs = input/l
            L = tf.linalg.diag(1 / self._feature_functions.kernel.lengthscales)  # [dim, dim]
            tf.debugging.assert_shapes([(self.noise_cov, (inputs.shape[-1], inputs.shape[-1]))])
            noise_cov = tf.convert_to_tensor(self.noise_cov, dtype=inputs.dtype)  # [dim, dim]
            # noise_cov = tf.linalg.diag(tf.cast(self.noise_cov, dtype=inputs.dtype))  # [dim, dim]
            x = inputs  # [N, dim]
            w = tf.transpose(self._feature_functions.W)  # [dim, M]
            Lw = tf.matmul(L, w)  # [dim, M]
            basis_functions = tf.cos(
                tf.transpose(tf.matmul(tf.transpose(Lw), x, transpose_b=True))
                + self._feature_functions.b
            )  # [N, M]
            # -------------------------------------------
            w_T_L_T_Sigma_Lw = tf.linalg.tensor_diag_part(
                tf.matmul(tf.matmul(Lw, noise_cov, transpose_a=True), Lw)
            )
            basis_functions = basis_functions * tf.math.exp(-w_T_L_T_Sigma_Lw / 2)[tf.newaxis, ...]
            # basis_functions = tf.matmul(basis_functions, # [M, M]
            #                             tf.math.exp(- tf.matmul(tf.matmul(Lw, noise_cov, transpose_a=True), Lw) / 2))

            output = c * basis_functions  # [N, M]
            # tf.ensure_shape(output, self._feature_functions.compute_output_shape(inputs.shape))
            return output
        elif self.noise_type == "uniform":
            # c remain unchanged
            c = tf.sqrt(
                2 * self._feature_functions.kernel.variance / self._feature_functions.n_components
            )
            # input: scaled_inputs = input/l
            L = tf.linalg.diag(1 / self._feature_functions.kernel.lengthscales)  # [dim, dim]
            x = inputs  # [N, dim]
            w = tf.transpose(self._feature_functions.W)  # [dim, M]
            Lw = tf.matmul(L, w)  # [dim, M]
            basis_functions = tf.cos(
                tf.transpose(tf.matmul(tf.transpose(Lw), x, transpose_b=True))
                + self._feature_functions.b
            )  # [N, M]
            output = c * basis_functions  # [N, M]

            w_L_delta = tf.matmul(self._feature_functions.W, L) * tf.cast(
                self.noise, inputs.dtype
            )  # [M, 1]
            # phi * prod(2\sqrt{pi/2}sinLδ/(wLδ))
            # output *= tf.reduce_prod((tf.sin(w_L_delta)) / (w_L_delta), axis=-1)
            intermediate = tf.sin(w_L_delta) / w_L_delta
            intermediate = tf.where(
                tf.math.is_nan(intermediate),
                tf.ones_like(intermediate),
                intermediate,
            )

            output *= tf.reduce_prod(intermediate, axis=-1)
            # tf.ensure_shape(output, self._feature_functions.compute_output_shape(inputs.shape))
            return output
        else:
            raise NotImplementedError("Only Uniform or Normal input uncertainty is supported atm")

    def feature_function_propagate_var(self, inputs):
        """
        calculate E[f^2]
        """
        if self.noise_type == "normal":
            c = self._feature_functions.kernel.variance / self._feature_functions.n_components

            # input: scaled_inputs = input/l
            L = tf.linalg.diag(1 / self._feature_functions.kernel.lengthscales)  # [dim, dim]
            tf.debugging.assert_shapes([(self.noise_cov, (inputs.shape[-1], inputs.shape[-1]))])
            noise_cov = tf.convert_to_tensor(self.noise_cov, dtype=inputs.dtype)  # [dim, dim]
            x = inputs  # [N, dim]

            # [M, dim] [M, dim] -> [M, M, dim]
            wm_plus_wn = (
                tf.expand_dims(self._feature_functions.W, -2) + self._feature_functions.W
            )  # [M, M, dim], symmteric
            wm_minus_wn = (
                tf.expand_dims(self._feature_functions.W, -2) - self._feature_functions.W
            )  # [M, M, dim]
            bm_plus_bn = (
                tf.transpose(self._feature_functions.b) + self._feature_functions.b
            )  # [M, M]
            bm_minus_bn = (
                tf.transpose(self._feature_functions.b) - self._feature_functions.b
            )  # [M, M]

            # calculate (w_m + w_n)Lx:
            # [M, M, 1, dim] @ [dim, dim] = [M, M, 1, dim]
            plus_intermediate = tf.matmul(tf.expand_dims(wm_plus_wn, -2), L, transpose_b=True)

            # FIXedME: Memory Consumption high: caused by [N, M, M]
            # [M, M, 1, dim] @ [dim, N] -> [M, M, 1, N] -> [M, M, N]
            wLx_plus = tf.squeeze(tf.matmul(plus_intermediate, x, transpose_b=True), axis=-2)
            wLx_plus = tf.transpose(wLx_plus, perm=[2, 0, 1])
            # [N, dim] @ [M, M, dim] -> [N, M, M]
            # wLx_plus = tf.reduce_sum(tf.expand_dims(tf.expand_dims(x, -1), -1) + plus_intermediate, axis=-1)

            # calculate (w_m + w_n)Lx:
            # [M, M, 1, dim]
            minus_intermediate = tf.matmul(tf.expand_dims(wm_minus_wn, -2), L, transpose_b=True)
            # minus_intermediate = tf.matmul(wm_minus_wn, L, transpose_b=True)  # [M, M, dim]
            wLx_minus = tf.squeeze(tf.matmul(minus_intermediate, x, transpose_b=True), axis=-2)
            wLx_minus = tf.transpose(wLx_minus, perm=[2, 0, 1])
            # wLx_minus = tf.reduce_sum(tf.expand_dims(tf.expand_dims(x, -1), -1) + minus_intermediate, axis=-1)

            # L^TΣL
            L_T_SigmaL = tf.matmul(tf.matmul(L, noise_cov, transpose_a=True), L)  # [dim, dim]

            # (w_m + w_n)^TL^TΣL(w_m + w_n)
            # [M ,M, 1, dim], [dim, dim], [M, M, dim] -> [M, M, 1, dim]
            N_f_plus = tf.matmul(tf.expand_dims(wm_plus_wn, -2), L_T_SigmaL)
            # [M, M, 1, dim] @ [M, M, dim, 1] -> [M, M]
            N_f_plus = tf.squeeze(tf.matmul(N_f_plus, tf.expand_dims(wm_plus_wn, -1)))

            # (w_m - w_n)^TL^TΣL(w_m - w_n)
            N_f_minus = tf.matmul(tf.expand_dims(wm_minus_wn, -2), L_T_SigmaL)  # [M, M, 1, dim]
            N_f_minus = tf.squeeze(tf.matmul(N_f_minus, tf.expand_dims(wm_minus_wn, -1)))  # [M, M]

            basis_functions = c * (
                tf.cos(wLx_plus + bm_plus_bn) * tf.math.exp(-N_f_plus / 2)
                + tf.cos(wLx_minus + bm_minus_bn) * tf.math.exp(-N_f_minus / 2)
            )
            output = basis_functions  # [N, M, M]
            return output

        elif self.noise_type == "uniform":
            # \sqrt(2σ^2/N_f)
            c = self._feature_functions.kernel.variance / self._feature_functions.n_components

            # input: scaled_inputs = input/l
            L = tf.linalg.diag(1 / self._feature_functions.kernel.lengthscales)  # [dim, dim]
            x = inputs  # [N, dim]
            # [M, dim] [M, dim] -> [M, M, dim]
            wm_plus_wn = (
                tf.expand_dims(self._feature_functions.W, -2) + self._feature_functions.W
            )  # [M, M, dim], symmteric
            wm_minus_wn = (
                tf.expand_dims(self._feature_functions.W, -2) - self._feature_functions.W
            )  # [M, M, dim]
            bm_plus_bn = (
                tf.transpose(self._feature_functions.b) + self._feature_functions.b
            )  # [M, M]
            bm_minus_bn = (
                tf.transpose(self._feature_functions.b) - self._feature_functions.b
            )  # [M, M]

            # calculate (w_m + w_n)Lx:
            # [M, M, 1, dim] @ [dim, dim] = [M, M, 1, dim]
            plus_intermediate = tf.matmul(tf.expand_dims(wm_plus_wn, -2), L, transpose_b=True)

            # [M, M, 1, dim] @ [dim, N] -> [M, M, 1， N] -> [M, M, N]
            wLx_plus = tf.squeeze(tf.matmul(plus_intermediate, x, transpose_b=True), axis=-2)
            wLx_plus = tf.transpose(wLx_plus, perm=[2, 0, 1])

            # [M, M, 1, dim]
            minus_intermediate = tf.matmul(tf.expand_dims(wm_minus_wn, -2), L, transpose_b=True)
            # minus_intermediate = tf.matmul(wm_minus_wn, L, transpose_b=True)  # [M, M, dim]
            wLx_minus = tf.squeeze(tf.matmul(minus_intermediate, x, transpose_b=True), axis=-2)
            wLx_minus = tf.transpose(wLx_minus, perm=[2, 0, 1])

            # [w_m + w_n]Lδ: [M, M, dim] @ [dim, dim] @ [dim, 1] = [M, M]
            wm_plus_wn_L_delta = tf.squeeze(
                tf.matmul(
                    tf.matmul(wm_plus_wn, L),
                    tf.cast(self.noise[tf.newaxis], dtype=wm_plus_wn.dtype),
                    transpose_b=True,
                ),
                -1,
            )  # [M, M]
            wm_minus_wn_L_delta = tf.squeeze(
                tf.matmul(
                    tf.matmul(wm_minus_wn, L),
                    tf.cast(self.noise[tf.newaxis], dtype=wm_minus_wn.dtype),
                    transpose_b=True,
                ),
                -1,
            )  # [M, M]

            # FIX NaN: Use x-> 0: sinx / x = 1
            sin_w_minus_L_delta_divided_w_L_delta = (
                tf.sin(wm_minus_wn_L_delta) / wm_minus_wn_L_delta
            )
            sin_w_minus_L_delta_divided_w_L_delta = tf.where(
                tf.math.is_nan(sin_w_minus_L_delta_divided_w_L_delta),
                tf.ones_like(sin_w_minus_L_delta_divided_w_L_delta),
                sin_w_minus_L_delta_divided_w_L_delta,
            )

            # [N, M, M]
            basis_functions = (
                c
                * tf.cos(wLx_plus + bm_plus_bn)
                * (tf.sin(wm_plus_wn_L_delta) / wm_plus_wn_L_delta)
                + c * tf.cos(wLx_minus + bm_minus_bn) * sin_w_minus_L_delta_divided_w_L_delta
            )
            output = basis_functions  # [N, M, M]
            return output
        else:
            raise NotImplementedError("Only Uniform or Normal input uncertainty is supported atm")

    def get_mean_trajectory(
        self,
        sample_size: int = 1,
        theta_sample: Optional[TensorType] = None,
        max_batch_element_num: Optional[int] = None,
        get_mean: Optional[bool] = False,
        return_sample: Optional[bool] = False,
        sample_method: str = "qMC",
    ) -> TrajectoryFunction:
        """
        Get RFF based mean sample trajectories
        :param sample_size: how many samples of mean trajectory is expected to be obtained
        :param theta_sample: if a theta_sample has been provided, use this theta sample to get the mean trajectory, used
            as a way of handling correlation
        :param max_batch_element_num: minibach sizes when evaluating mean trajectories, used to avoid OOM issue
        :param get_mean used when only want the mean of the mean trajectory
        :param return_sample return theta samples used to generate current trajectories
        """
        # input check
        if sample_size != 1:
            assert theta_sample is None
            assert get_mean is False
        if theta_sample is not None:
            assert get_mean is False

        if get_mean:
            theta_sample = self._theta_posterior.loc[tf.newaxis]
        else:
            theta_sample = (
                self._theta_posterior.sample(sample_size, sample_method)
                if theta_sample is None
                else theta_sample
            )  # [1, m]

        @sequential_batch(max_batch_element_num)
        def trajectory(x: TensorType) -> TensorType:
            feature_evaluations = self.feature_function_propagate_mean(x)  # [N, m]
            # [sample_size, m/2m] * [m/2m, N] -> [sample_size, N] -> [sample_size, N, 1]
            return tf.matmul(theta_sample, feature_evaluations, transpose_b=True)[..., tf.newaxis]

        return trajectory if not return_sample else (trajectory, theta_sample)

    def get_var_trajectory(
        self,
        sample_size: int = 1,
        theta_sample: Optional[TensorType] = None,
        max_batch_element_num: Optional[int] = None,
        get_mean: bool = False,
        sample_method: str = "qMC",
    ) -> TrajectoryFunction:
        """
        :param sample_size
        :param theta_sample
        :param max_batch_element_num
        :param get_mean
        """
        # input check
        if sample_size != 1:
            assert theta_sample is None
            assert get_mean is False
        if theta_sample is not None:
            assert get_mean is False

        if get_mean is True:  # return the mean of variance distribution
            theta_mean = self._theta_posterior.loc[tf.newaxis]
            theta_cov = self._theta_posterior.cov

            @sequential_batch(max_batch_element_num)
            def mean_trajectory(x: TensorType) -> TensorType:
                """
                calculate E[f^2] - E[f]^2 = E[f^2] - J^2
                """
                mean_feature_evaluations = self.feature_function_propagate_mean(x)
                J = tf.matmul(mean_feature_evaluations, theta_mean, transpose_b=True)  # []
                # fisrt part: L^{-1}sum([(Lμ)_i^2 + (LΣL^T)_{i, i}])
                var_feature_evaluations = self.feature_function_propagate_var(x)  # [N, M, M]
                first_first_part = tf.squeeze(
                    tf.linalg.trace(tf.matmul(var_feature_evaluations, theta_cov))
                )
                first_second_part = tf.squeeze(
                    tf.matmul(
                        tf.matmul(theta_mean, var_feature_evaluations), theta_mean, transpose_b=True
                    )
                )  # N
                first_part = tf.expand_dims(first_first_part + first_second_part, -1)

                second_part = J ** 2 + tf.squeeze(
                    tf.matmul(
                        tf.matmul(
                            tf.expand_dims(mean_feature_evaluations, -1),
                            theta_cov,
                            transpose_a=True,
                        ),
                        tf.expand_dims(mean_feature_evaluations, -1),
                    ),
                    -1,
                )
                return (first_part - second_part)[tf.newaxis, ...]  # [1,N,1]

            return mean_trajectory

        else:
            theta_sample = (
                self._theta_posterior.sample(sample_size, sample_method)
                if theta_sample is None
                else theta_sample
            )  # [1, m]

            @sequential_batch(max_batch_element_num)
            def trajectory(x: TensorType) -> TensorType:
                """
                calculate E[f^2] - E[f]^2 = E[f^2] - J^2
                """
                mean_feature_evaluations = self.feature_function_propagate_mean(x)
                # [sample_size, N, 1]
                J = tf.matmul(theta_sample, mean_feature_evaluations, transpose_b=True)[
                    ..., tf.newaxis
                ]
                var_feature_evaluations = self.feature_function_propagate_var(x)  # [N, M, M]
                # θ^T * [N, M, M] * θ
                # expand theta_sample:
                expand_theta_sample = tf.expand_dims(
                    tf.expand_dims(theta_sample, -2), -2
                )  # [sample_size, 1, 1, M]
                first_part = tf.matmul(
                    expand_theta_sample, var_feature_evaluations
                )  # [N, MC_num, M]
                second_part = tf.reduce_sum(
                    tf.multiply(first_part, expand_theta_sample), axis=-1
                )  # [N, MC_num]
                E_f_sqaure = second_part
                return E_f_sqaure - J ** 2  # [N,MC_size]

            return trajectory


class QFFVarianceSampler(RandomFourierFeatureThompsonSampler, VarianceSampler):
    def __init__(
        self,
        noise_type: str,
        model: ProbabilisticModel,
        dataset: Dataset,
        sample_min_value: bool = False,
        num_features: int = 256,
        **kwargs,
    ):
        assert noise_type in ["normal", "uniform"]
        if noise_type == "normal":
            # scale the normal properly
            self.noise_type = noise_type
            self.noise_cov = kwargs["noise"]
        else:
            assert noise_type == "uniform"
            self.noise_type = noise_type
            self.noise = kwargs["noise"]
            tf.debugging.assert_shapes([(self.noise, ("D"))])
        super().__init__(
            1, model, dataset, sample_min_value, num_features, fourier_feature_method="qff"
        )

    def feature_function_propagate_mean(self, inputs):
        """
        calculate feature function propagated by mean
        """
        if self.noise_type == "normal":
            # c remain unchanged
            # input: scaled_inputs = input/l
            L = tf.linalg.diag(1 / self._feature_functions.kernel.lengthscales)  # [dim, dim]
            tf.debugging.assert_shapes([(self.noise_cov, (inputs.shape[-1], inputs.shape[-1]))])
            noise_cov = tf.convert_to_tensor(self.noise_cov, dtype=inputs.dtype)  # [dim, dim]

            w = tf.transpose(self._feature_functions.abscissa)  # [dim, Nf^D]
            Lw = tf.matmul(L, w)  # [dim, M]
            exp_w_L_cov_L_w = tf.linalg.diag_part(
                tf.math.exp(-tf.matmul(tf.matmul(Lw, noise_cov, transpose_a=True), Lw) / 2)
            )  # [N_f]

            # basis_functions = self._feature_functions(x) # [N, 2Nf]
            Lx = tf.divide(inputs, self._feature_functions.kernel.lengthscales)  # [N, D]
            integral_weights = self._feature_functions._compute_constant()  # [2m]
            wLx = self._feature_functions._compute_bases(
                Lx
            )  # [cos(wLx), sin(wLx)]
            basis_functions = integral_weights * wLx  # omega * [cos(wLx), sin(wLx) [Nf] * [N, Nf]
            basis_functions = basis_functions * tf.tile(exp_w_L_cov_L_w, [2])
            output = basis_functions  # [N, M]
            # tf.ensure_shape(output, self._feature_functions.compute_output_shape(inputs.shape))
            return output
        elif self.noise_type == "uniform":
            L = tf.linalg.diag(1 / self._feature_functions.kernel.lengthscales)  # [dim, dim]

            w = tf.transpose(self._feature_functions.abscissa)  # [dim, Nf^D]
            w_L_delta = tf.matmul(w, L, transpose_a=True) * tf.cast(
                self.noise, inputs.dtype
            )  # [M, 1]

            # basis_functions = self._feature_functions(x) # [N, 2Nf]
            Lx = tf.divide(inputs, self._feature_functions.kernel.lengthscales)  # [N, D]
            integral_weights = self._feature_functions._compute_constant()  # [2m] weight
            wLx = self._feature_functions._compute_bases(
                Lx
            )  # [sin(wLx), cos(wLx)], w is the node of the spectral density
            basis_functions = integral_weights * wLx  # omega * [sin(wLx), cos(wLx)] [Nf] * [N, Nf]
            intermediate = tf.sin(w_L_delta) / w_L_delta
            intermediate = tf.where(
                tf.math.is_nan(intermediate),
                tf.ones_like(intermediate),
                intermediate,
            )
            basis_functions *= tf.tile(tf.reduce_prod(intermediate, axis=-1), [2])
            output = basis_functions  # [N, M]
            return output
        else:
            raise NotImplementedError("Only Uniform or Normal input uncertainty is supported atm")

    def feature_function_propagate_var(self, inputs):
        """
        calculate E[f^2]
        """
        if self.noise_type == "normal":
            constant_factors = tf.sqrt(self._feature_functions.factors)[
                ..., tf.newaxis
            ]  # quadrature weight
            c = (
                tf.matmul(constant_factors, constant_factors, transpose_b=True)
                * self._feature_functions.kernel.variance
                / 2
            )
            # input: scaled_inputs = input/l
            L = tf.linalg.diag(1 / self._feature_functions.kernel.lengthscales)  # [dim, dim]
            # Lx = tf.divide(inputs, self._feature_functions.kernel.lengthscales)
            tf.debugging.assert_shapes([(self.noise_cov, (inputs.shape[-1], inputs.shape[-1]))])
            noise_cov = tf.convert_to_tensor(self.noise_cov, dtype=inputs.dtype)  # [dim, dim]
            x = inputs  # [N, dim]
            W = self._feature_functions.abscissa  # [dim, Nf^D]
            # [M, dim] [M, dim] -> [M, M, dim]
            wm_plus_wn = tf.expand_dims(W, -2) + W  # [M, M, dim], symmteric

            wm_minus_wn = tf.expand_dims(W, -2) - W  # [M, M, dim]
            minus_wm_plus_wn = tf.expand_dims(-W, -2) + W  # [M, M, dim]
            # calculate (w_m + w_n)Lx:
            # [M, M, 1, dim] @ [dim, dim] = [M, M, 1, dim]
            plus_intermediate = tf.matmul(tf.expand_dims(wm_plus_wn, -2), L, transpose_b=True)
            # [M, M, 1, dim] @ [dim, N] -> [M, M, 1, N] -> [M, M, N]
            wm_plus_wn_L_x = tf.squeeze(tf.matmul(plus_intermediate, x, transpose_b=True), axis=-2)
            wm_plus_wn_L_x = tf.transpose(wm_plus_wn_L_x, perm=[2, 0, 1])  # [N, M, M]
            # calculate (w_m - w_n)Lx:
            # [M, M, 1, dim]
            minus_intermediate = tf.matmul(tf.expand_dims(wm_minus_wn, -2), L, transpose_b=True)
            wm_minus_wn_L_x = tf.squeeze(
                tf.matmul(minus_intermediate, x, transpose_b=True), axis=-2
            )
            wm_minus_wn_L_x = tf.transpose(wm_minus_wn_L_x, perm=[2, 0, 1])  # [N, M, M]
            # L^TΣL
            L_T_Sigma_L = tf.matmul(tf.matmul(L, noise_cov, transpose_a=True), L)  # [dim, dim]

            # (w_m + w_n)^TL^TΣL(w_m + w_n)
            # [M ,M, 1, dim], [dim, dim], [M, M, dim] -> [M, M, 1, dim]
            wm_wn_L_Sigma_L_wm_wn = tf.matmul(tf.expand_dims(wm_plus_wn, -2), L_T_Sigma_L)
            # [M, M, 1, dim] @ [M, M, dim, 1] -> [M, M]
            wm_wn_L_Sigma_L_wm_wn = tf.squeeze(
                tf.matmul(wm_wn_L_Sigma_L_wm_wn, tf.expand_dims(wm_plus_wn, -1))
            )

            # (w_m - w_n)^TL^TΣL(w_m - w_n)
            wm_minus_wn_L_Sigma_L_wm_minus_wn = tf.matmul(
                tf.expand_dims(wm_minus_wn, -2), L_T_Sigma_L
            )  # [M, M, 1, dim]
            wm_minus_wn_L_Sigma_L_wm_minus_wn = tf.squeeze(
                tf.matmul(wm_minus_wn_L_Sigma_L_wm_minus_wn, tf.expand_dims(wm_minus_wn, -1))
            )  # [M, M]
            block_basis_func1 = c * (
                tf.cos(wm_plus_wn_L_x) * tf.math.exp(-wm_wn_L_Sigma_L_wm_wn / 2)
                + tf.cos(wm_minus_wn_L_x) * tf.math.exp(-wm_minus_wn_L_Sigma_L_wm_minus_wn / 2)
            )  # [N, M, M]
            block_basis_func2 = c * (
                tf.sin(wm_plus_wn_L_x) * tf.math.exp(-wm_wn_L_Sigma_L_wm_wn / 2)
                + tf.sin(wm_minus_wn_L_x) * tf.math.exp(-wm_minus_wn_L_Sigma_L_wm_minus_wn / 2)
            )  # [N, M, M]

            block_basis_func3 = tf.transpose(block_basis_func2, perm=[0, 2, 1])
            block_basis_func4 = c * (
                tf.cos(wm_minus_wn_L_x) * tf.math.exp(-wm_minus_wn_L_Sigma_L_wm_minus_wn / 2)
                - tf.cos(wm_plus_wn_L_x) * tf.math.exp(-wm_wn_L_Sigma_L_wm_wn / 2)
            )
            # merge block matrix
            _inter1 = tf.concat([block_basis_func1, block_basis_func2], axis=-2)
            _inter2 = tf.concat([block_basis_func3, block_basis_func4], axis=-2)
            basis_functions = tf.concat([_inter1, _inter2], axis=-1)
            output = basis_functions  # [N, M, M]
            return output

        elif self.noise_type == "uniform":
            constant_factors = tf.sqrt(self._feature_functions.factors)[
                ..., tf.newaxis
            ]  # quadrature weight
            c = (
                tf.matmul(constant_factors, constant_factors, transpose_b=True)
                * self._feature_functions.kernel.variance
                / 2
            )
            # input: scaled_inputs = input/l
            L = tf.linalg.diag(1 / self._feature_functions.kernel.lengthscales)  # [dim, dim]

            x = inputs  # [N, dim]
            W = self._feature_functions.abscissa  # [dim, Nf^D]

            # [M, dim] [M, dim] -> [M, M, dim]
            wm_plus_wn = tf.expand_dims(W, -2) + W  # [M, M, dim], symmteric

            wm_minus_wn = tf.expand_dims(W, -2) - W  # [M, M, dim]
            minus_wm_plus_wn = tf.expand_dims(-W, -2) + W  # [M, M, dim]

            plus_intermediate = tf.matmul(tf.expand_dims(wm_plus_wn, -2), L, transpose_b=True)

            # [M, M, 1, dim] @ [dim, N] -> [M, M, 1, N] -> [M, M, C]
            wm_plus_wn_Lx = tf.squeeze(tf.matmul(plus_intermediate, x, transpose_b=True), axis=-2)
            wm_plus_wn_Lx = tf.transpose(wm_plus_wn_Lx, perm=[2, 0, 1])
            # [M, M, 1, dim]
            minus_intermediate = tf.matmul(tf.expand_dims(wm_minus_wn, -2), L, transpose_b=True)
            # minus_intermediate = tf.matmul(wm_minus_wn, L, transpose_b=True)  # [M, M, dim]
            wm_minus_wn_Lx = tf.squeeze(tf.matmul(minus_intermediate, x, transpose_b=True), axis=-2)
            wm_minus_wn_Lx = tf.transpose(wm_minus_wn_Lx, perm=[2, 0, 1])
            minus_wm_plus_wn_L = tf.matmul(
                tf.expand_dims(minus_wm_plus_wn, -2), L, transpose_b=True
            )
            minus_wm_plus_wn_Lx = tf.squeeze(
                tf.matmul(minus_wm_plus_wn_L, x, transpose_b=True), axis=-2
            )
            minus_wm_plus_wn_Lx = tf.transpose(minus_wm_plus_wn_Lx, perm=[2, 0, 1])
            wm_plus_wn_L_delta = tf.squeeze(
                tf.matmul(
                    tf.matmul(wm_plus_wn, L),
                    tf.cast(self.noise[tf.newaxis], dtype=wm_plus_wn.dtype),
                    transpose_b=True,
                ),
                -1,
            )  # [M, M]
            minus_wm_plus_wn_L_delta = tf.squeeze(
                tf.matmul(
                    tf.matmul(minus_wm_plus_wn, L),
                    tf.cast(self.noise[tf.newaxis], dtype=minus_wm_plus_wn.dtype),
                    transpose_b=True,
                ),
                -1,
            )  # [M, M]
            wm_minus_wn_L_delta = tf.squeeze(
                tf.matmul(
                    tf.matmul(wm_minus_wn, L),
                    tf.cast(self.noise[tf.newaxis], dtype=wm_minus_wn.dtype),
                    transpose_b=True,
                ),
                -1,
            )  # [M, M]

            sin_w_plus_L_delta_divided_w_L_delta = tf.sin(wm_plus_wn_L_delta) / wm_plus_wn_L_delta
            sin_w_plus_L_delta_divided_w_L_delta = tf.where(
                tf.math.is_nan(sin_w_plus_L_delta_divided_w_L_delta),
                tf.ones_like(sin_w_plus_L_delta_divided_w_L_delta),
                sin_w_plus_L_delta_divided_w_L_delta,
            )

            sin_w_minus_L_delta_divided_w_L_delta = (
                tf.sin(wm_minus_wn_L_delta) / wm_minus_wn_L_delta
            )
            sin_w_minus_L_delta_divided_w_L_delta = tf.where(
                tf.math.is_nan(sin_w_minus_L_delta_divided_w_L_delta),
                tf.ones_like(sin_w_minus_L_delta_divided_w_L_delta),
                sin_w_minus_L_delta_divided_w_L_delta,
            )

            block_basis_func1 = (
                c * tf.cos(wm_plus_wn_Lx) * sin_w_plus_L_delta_divided_w_L_delta
                + c * tf.cos(wm_minus_wn_Lx) * sin_w_minus_L_delta_divided_w_L_delta
            )
            block_basis_func2 = (
                c * tf.sin(wm_plus_wn_Lx) * sin_w_plus_L_delta_divided_w_L_delta
                + c * tf.sin(wm_minus_wn_Lx) * sin_w_minus_L_delta_divided_w_L_delta
            )

            sin_minus_w_L_delta_divided_w_L_delta = (
                tf.sin(minus_wm_plus_wn_L_delta) / minus_wm_plus_wn_L_delta
            )
            sin_minus_w_L_delta_divided_w_L_delta = tf.where(
                tf.math.is_nan(sin_minus_w_L_delta_divided_w_L_delta),
                tf.ones_like(sin_minus_w_L_delta_divided_w_L_delta),
                sin_minus_w_L_delta_divided_w_L_delta,
            )
            block_basis_func3 = (
                c * tf.sin(wm_plus_wn_Lx) * sin_w_plus_L_delta_divided_w_L_delta
                + c * tf.sin(minus_wm_plus_wn_Lx) * sin_minus_w_L_delta_divided_w_L_delta
            )

            block_basis_func4 = c * (
                tf.cos(wm_minus_wn_Lx) * sin_w_minus_L_delta_divided_w_L_delta
                - tf.cos(wm_plus_wn_Lx) * sin_w_plus_L_delta_divided_w_L_delta
            )
            # merge block matrix
            _inter1 = tf.concat([block_basis_func1, block_basis_func2], axis=-2)
            _inter2 = tf.concat([block_basis_func3, block_basis_func4], axis=-2)
            basis_functions = tf.concat([_inter1, _inter2], axis=-1)
            output = basis_functions  # [N, M, M]
            return output
        else:
            raise NotImplementedError("Only Uniform or Normal input uncertainty is supported atm")

    def get_mean_trajectory(
        self,
        sample_size: int = 1,
        theta_sample: TensorType = None,
        max_batch_element_num=None,
        get_mean: bool = False,
        return_sample: bool = False,
        sample_method: str = "qMC",
    ) -> TrajectoryFunction:
        """
        Get QFF based mean sample trajectories

        :param sample_size: how many samples of mean trajectory is expected to be obtained
        :param theta_sample: if a theta_sample has been provided, use this theta sample to get the mean trajectory, used
            as a way of handling correlation
        :param max_batch_element_num: minibach sizes when evaluating mean trajectories, used to avoid OOM issue
        :param get_mean used when only want the mean of the mean trajectory
        :param return_sample return theta samples used to generate current trajectories
        """
        # input check
        if sample_size != 1:
            assert theta_sample is None
            assert get_mean is False
        if theta_sample is not None:
            assert get_mean is False

        if get_mean:
            theta_sample = self._theta_posterior.loc[tf.newaxis]
        else:
            theta_sample = (
                self._theta_posterior.sample(sample_size, sample_method)
                if theta_sample is None
                else theta_sample
            )  # [1, 2m]

        @sequential_batch(max_batch_element_num)
        def trajectory(x: TensorType) -> TensorType:
            feature_evaluations = self.feature_function_propagate_mean(x)  # [N, 2m]
            return tf.matmul(theta_sample, feature_evaluations, transpose_b=True)[..., tf.newaxis]

        return trajectory if not return_sample else (trajectory, theta_sample)

    def get_var_trajectory(
        self,
        sample_size=1,
        theta_sample: TensorType = None,
        max_batch_element_num=None,
        get_mean: bool = False,
        sample_method: str = "qMC",
    ) -> TrajectoryFunction:
        if get_mean is True:  # return the mean of variance distribution
            theta_mean = self._theta_posterior.loc[tf.newaxis]
            theta_cov = self._theta_posterior.cov

            @sequential_batch(max_batch_element_num)
            def trajectory(x: TensorType) -> TensorType:
                """
                calculate E[f^2] - E[f]^2 = E[f^2] - J^2
                """
                mean_feature_evaluations = self.feature_function_propagate_mean(x)
                J = tf.matmul(mean_feature_evaluations, theta_mean, transpose_b=True)  # []
                # fisrt part: L^{-1}sum([(Lμ)_i^2 + (LΣL^T)_{i, i}])
                var_feature_evaluations = self.feature_function_propagate_var(x)  # [N, M, M]
                first_first_part = tf.squeeze(
                    tf.linalg.trace(tf.matmul(var_feature_evaluations, theta_cov))
                )
                first_second_part = tf.squeeze(
                    tf.matmul(
                        tf.matmul(theta_mean, var_feature_evaluations), theta_mean, transpose_b=True
                    )
                )  # N
                first_part = tf.expand_dims(first_first_part + first_second_part, -1)

                second_part = J ** 2 + tf.squeeze(
                    tf.matmul(
                        tf.matmul(
                            tf.expand_dims(mean_feature_evaluations, -1),
                            theta_cov,
                            transpose_a=True,
                        ),
                        tf.expand_dims(mean_feature_evaluations, -1),
                    ),
                    -1,
                )
                return (first_part - second_part)[tf.newaxis, ...]  # [1, N, 1]

            return trajectory
        else:
            theta_sample = (
                self._theta_posterior.sample(sample_size, sample_method)
                if theta_sample is None
                else theta_sample
            )  # [sample_size, m]

            @sequential_batch(max_batch_element_num)
            def trajectory(x: TensorType) -> TensorType:
                """
                calculate E[f^2] - E[f]^2 = E[f^2] - J^2
                """
                mean_feature_evaluations = self.feature_function_propagate_mean(x)
                # [sample_size, N, 1]
                J = tf.matmul(theta_sample, mean_feature_evaluations, transpose_b=True)[
                    ..., tf.newaxis
                ]
                var_feature_evaluations = self.feature_function_propagate_var(x)  # [N, M, M]
                # θ^T * [N, M, M] * θ
                # expand theta_sample:
                expand_theta_sample = tf.expand_dims(
                    tf.expand_dims(theta_sample, -2), -2
                )  # [sample_size, 1, 1, M]
                first_part = tf.matmul(
                    expand_theta_sample, var_feature_evaluations
                )  # [N, MC_num, M]
                second_part = tf.reduce_sum(
                    tf.multiply(first_part, expand_theta_sample), axis=-1
                )  # [N, MC_num]
                E_f_sqaure = second_part
                return E_f_sqaure - J ** 2  # [N,MC_size]

            return trajectory
