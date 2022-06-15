import gpflow
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import assert_datasets_allclose, quadratic, random_seed
from trieste.models.rmo_models.kernels.noise_kernels import (
    NoisyAnalyticSquaredExponential,
    NoisyMCSquaredExponential,
)
from trieste.models.rmo_models.noise_GP import NGPRAnalyticKernel, NGPRMCKernel
from trieste.type import TensorType
from trieste.utils.objectives import branin


def h(x):
    return -(
        2 - 0.8 * tf.exp(-(((x - 0.35) / 0.25) ** 2)) - tf.exp(-(((x - 0.8) / 0.05) ** 2)) - 1.5
    )


_1dx = tf.cast(tf.linspace([0], [1], 35), dtype=tf.float64)
_1dy = h(_1dx)

_2dx = tf.constant(
    [
        [0.2, 0.4],
        [0.4, 0.1],
        [0.23, 0.9],
        [0.8, 0.4],
        [0.6, 0.1],
        [0.9, 0.2],
        [0.6, 0.2],
        [0.1, 0.3],
    ],
    dtype=tf.float64,
)
_2dy = branin(_2dx)


@random_seed
@pytest.mark.parametrize(
    "training_data, kernel, mc_kernel, test_x",
    [
        (
            (_1dx, _1dy),
            NoisyAnalyticSquaredExponential(
                lengthscales=[0.1], noise_covar=tf.constant([0.001], dtype=tf.float64)
            ),
            NoisyMCSquaredExponential(
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(5000),
                    dtype=tf.float64,
                ),
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(5000),
                    dtype=tf.float64,
                ),
            ),
            tf.constant([[0.22], [0.54], [0.92], [0.31], [0.71]], dtype=tf.float64),
        ),
        (
            (_2dx, _2dy),
            NoisyAnalyticSquaredExponential(
                lengthscales=[0.1] * 2, noise_covar=tf.constant([0.001] * 2, dtype=tf.float64)
            ),
            NoisyMCSquaredExponential(
                tf.cast(
                    tfp.distributions.Normal(loc=[[0] * 2], scale=tf.sqrt([[0.001] * 2])).sample(
                        5000
                    ),
                    dtype=tf.float64,
                ),
                tf.cast(
                    tfp.distributions.Normal(loc=[[0] * 2], scale=tf.sqrt([[0.001] * 2])).sample(
                        5000
                    ),
                    dtype=tf.float64,
                ),
            ),
            tf.constant(
                [[0.22, 0.71], [0.54, 0.67], [0.92, 0.18], [0.31, 0.55], [0.71, 0.29]],
                dtype=tf.float64,
            ),
        ),
    ],
)
def test_K(
    training_data: tuple,
    kernel,
    mc_kernel,
    test_x: TensorType,
) -> None:
    ngpr = NGPRAnalyticKernel(training_data, kernel)
    opt = gpflow.optimizers.Scipy()
    _ = opt.minimize(ngpr.training_loss, ngpr.trainable_variables, options=dict(maxiter=10))
    #
    nmcgpr = NGPRMCKernel(training_data, mc_kernel)
    # assign kernel hyper param
    nmcgpr.kernel.lengthscales = ngpr.kernel.lengthscales
    nmcgpr.kernel.variance = ngpr.kernel.variance
    nmcgpr.likelihood.variance = ngpr.likelihood.variance

    K_mm_analytic = ngpr.kernel(test_x)
    K_mm_MC = nmcgpr.kernel(test_x)
    tf.debugging.assert_near(K_mm_analytic, K_mm_MC, rtol=1e-7, atol=1e-7)


@random_seed
@pytest.mark.parametrize(
    "training_data, kernel, mc_kernel, test_x, full_cov, rtol",
    [
        (
            (_1dx, _1dy),
            NoisyAnalyticSquaredExponential(
                lengthscales=[0.1], noise_covar=tf.constant([0.001], dtype=tf.float64)
            ),
            NoisyMCSquaredExponential(
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(50000),
                    dtype=tf.float64,
                ),
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(50000),
                    dtype=tf.float64,
                ),
            ),
            tf.constant([[0.22], [0.54], [0.92], [0.31], [0.71]], dtype=tf.float64),
            False,
            5e-3,
        ),
        (
            (_1dx, _1dy),
            NoisyAnalyticSquaredExponential(
                lengthscales=[0.1], noise_covar=tf.constant([0.001], dtype=tf.float64)
            ),
            NoisyMCSquaredExponential(
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(5000),
                    dtype=tf.float64,
                ),
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(5000),
                    dtype=tf.float64,
                ),
            ),
            tf.constant(
                [
                    [0.22],
                    [0.25],
                    [0.28],
                    [0.35],
                    [0.39],
                    [0.42],
                    [0.47],
                    [0.66],
                    [0.81],
                    [0.54],
                    [0.92],
                    [0.31],
                    [0.71],
                ],
                dtype=tf.float64,
            ),
            True,
            1e-1,
        ),
        (
            (_2dx, _2dy),
            NoisyAnalyticSquaredExponential(
                lengthscales=[0.1] * 2, noise_covar=tf.constant([0.001] * 2, dtype=tf.float64)
            ),
            NoisyMCSquaredExponential(
                tf.cast(
                    tfp.distributions.Normal(loc=[[0] * 2], scale=tf.sqrt([[0.001] * 2])).sample(
                        5000
                    ),
                    dtype=tf.float64,
                ),
                tf.cast(
                    tfp.distributions.Normal(loc=[[0] * 2], scale=tf.sqrt([[0.001] * 2])).sample(
                        5000
                    ),
                    dtype=tf.float64,
                ),
            ),
            tf.constant(
                [[0.22, 0.71], [0.54, 0.67], [0.92, 0.18], [0.31, 0.55], [0.71, 0.29]],
                dtype=tf.float64,
            ),
            True,
            1e-2,
        ),
    ],
)
def test_K_JJ(
    training_data: tuple, kernel, mc_kernel, test_x: TensorType, full_cov: bool, rtol: TensorType
) -> None:
    ngpr = NGPRAnalyticKernel(training_data, kernel)
    opt = gpflow.optimizers.Scipy()
    _ = opt.minimize(ngpr.training_loss, ngpr.trainable_variables, options=dict(maxiter=10))
    #
    nmcgpr = NGPRMCKernel(training_data, mc_kernel)
    # assign kernel hyper param
    nmcgpr.kernel.lengthscales = ngpr.kernel.lengthscales
    nmcgpr.kernel.variance = ngpr.kernel.variance
    nmcgpr.likelihood.variance = ngpr.likelihood.variance

    K_mm_analytic = ngpr.kernel.Kgg(test_x, full_cov=full_cov)
    K_mm_MC = nmcgpr.kernel.Kgg(test_x, full_cov=full_cov)
    if full_cov is True:
        tf.debugging.assert_near(
            K_mm_analytic, tf.reduce_mean(K_mm_MC, axis=-3), rtol=rtol, atol=1e-4
        )
    else:
        tf.debugging.assert_near(K_mm_analytic, tf.reduce_mean(K_mm_MC, axis=-2), rtol=rtol)


@random_seed
@pytest.mark.parametrize(
    "training_data, kernel, mc_kernel, test_x, rtol",
    [
        (
            (_1dx, _1dy),
            NoisyAnalyticSquaredExponential(
                lengthscales=[0.1], noise_covar=tf.constant([0.001], dtype=tf.float64)
            ),
            NoisyMCSquaredExponential(
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(50000),
                    dtype=tf.float64,
                ),
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(50000),
                    dtype=tf.float64,
                ),
            ),
            tf.constant([[0.22], [0.54], [0.92], [0.31], [0.71]], dtype=tf.float64),
            1e-2,
        ),
        (
            (_2dx, _2dy),
            NoisyAnalyticSquaredExponential(
                lengthscales=[0.1] * 2, noise_covar=tf.constant([0.001] * 2, dtype=tf.float64)
            ),
            NoisyMCSquaredExponential(
                tf.cast(
                    tfp.distributions.Normal(loc=[[0] * 2], scale=tf.sqrt([[0.001] * 2])).sample(
                        5000
                    ),
                    dtype=tf.float64,
                ),
                tf.cast(
                    tfp.distributions.Normal(loc=[[0] * 2], scale=tf.sqrt([[0.001] * 2])).sample(
                        5000
                    ),
                    dtype=tf.float64,
                ),
            ),
            tf.constant(
                [[0.22, 0.71], [0.54, 0.67], [0.92, 0.18], [0.31, 0.55], [0.71, 0.29]],
                dtype=tf.float64,
            ),
            1e-2,
        ),
    ],
)
def test_K_Jf(
    training_data: tuple, kernel, mc_kernel, test_x: TensorType, rtol: TensorType
) -> None:
    ngpr = NGPRAnalyticKernel(training_data, kernel)
    opt = gpflow.optimizers.Scipy()
    _ = opt.minimize(ngpr.training_loss, ngpr.trainable_variables, options=dict(maxiter=10))
    #
    nmcgpr = NGPRMCKernel(training_data, mc_kernel)
    # assign kernel hyper param
    nmcgpr.kernel.lengthscales = ngpr.kernel.lengthscales
    nmcgpr.kernel.variance = ngpr.kernel.variance
    nmcgpr.likelihood.variance = ngpr.likelihood.variance

    K_mm_analytic = ngpr.kernel.Kgf(training_data[0], test_x)
    K_mm_MC = nmcgpr.kernel.Kgf(training_data[0], test_x)
    tf.debugging.assert_near(K_mm_analytic, tf.reduce_mean(K_mm_MC, axis=-3), rtol=rtol, atol=1e-5)


@random_seed
@pytest.mark.parametrize(
    "training_data, kernel, mc_kernel, test_x, rtol",
    [
        (
            (_1dx, _1dy),
            NoisyAnalyticSquaredExponential(
                lengthscales=[0.1], noise_covar=tf.constant([0.001], dtype=tf.float64)
            ),
            NoisyMCSquaredExponential(
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(50000),
                    dtype=tf.float64,
                ),
                tf.cast(
                    tfp.distributions.Normal(loc=[[0]], scale=tf.sqrt([[0.001]])).sample(50000),
                    dtype=tf.float64,
                ),
            ),
            tf.constant([[0.22], [0.54], [0.92], [0.31], [0.71]], dtype=tf.float64),
            1e-2,
        ),
        (
            (_2dx, _2dy),
            NoisyAnalyticSquaredExponential(
                lengthscales=[0.1] * 2, noise_covar=tf.constant([0.001] * 2, dtype=tf.float64)
            ),
            NoisyMCSquaredExponential(
                tf.cast(
                    tfp.distributions.Normal(loc=[[0] * 2], scale=tf.sqrt([[0.001] * 2])).sample(
                        5000
                    ),
                    dtype=tf.float64,
                ),
                tf.cast(
                    tfp.distributions.Normal(loc=[[0] * 2], scale=tf.sqrt([[0.001] * 2])).sample(
                        5000
                    ),
                    dtype=tf.float64,
                ),
            ),
            tf.constant(
                [[0.22, 0.71], [0.54, 0.67], [0.92, 0.18], [0.31, 0.55], [0.71, 0.29]],
                dtype=tf.float64,
            ),
            1e-2,
        ),
    ],
)
def test_K_fJ(
    training_data: tuple, kernel, mc_kernel, test_x: TensorType, rtol: TensorType
) -> None:
    ngpr = NGPRAnalyticKernel(training_data, kernel)
    opt = gpflow.optimizers.Scipy()
    _ = opt.minimize(ngpr.training_loss, ngpr.trainable_variables, options=dict(maxiter=10))
    #
    nmcgpr = NGPRMCKernel(training_data, mc_kernel)
    # assign kernel hyper param
    nmcgpr.kernel.lengthscales = ngpr.kernel.lengthscales
    nmcgpr.kernel.variance = ngpr.kernel.variance
    nmcgpr.likelihood.variance = ngpr.likelihood.variance

    K_mm_analytic = tf.linalg.matrix_transpose(ngpr.kernel.Kgf(test_x, training_data[0]))
    K_mm_MC = nmcgpr.kernel.Kfg(training_data[0], test_x)
    tf.debugging.assert_near(K_mm_analytic, tf.reduce_mean(K_mm_MC, axis=-3), rtol=rtol, atol=1e-5)
