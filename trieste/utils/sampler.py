import math

import tensorflow as tf

from ..space import Box
from ..types import TensorType


class QuasiMonteCarloNormalSampler:
    """
    Sample from a univariate normal:
    N~(0, I_d), where d is the dimensionality
    """

    def __init__(self, dimensionality: int):
        self.dimensionality = dimensionality
        self._box_muller_req_dim = 2 * math.ceil(
            dimensionality / 2
        )  # making sure this dim is even number
        self.dimensionality = dimensionality
        self._uniform_engine = Box(
            tf.zeros(shape=self._box_muller_req_dim), tf.ones(shape=self._box_muller_req_dim)
        )

    def sample(self, sample_size: int):
        """
        main reference:
        """
        uniform_samples = self._uniform_engine.sample_halton(
            sample_size
        )  # [sample_size, ceil_even(dimensionality)]

        even_indices = tf.range(0, uniform_samples.shape[-1], 2)
        Rs = tf.sqrt(
            -2 * tf.math.log(tf.gather(uniform_samples, even_indices, axis=-1))
        )  # [sample_size, ceil_even(dimensionality)/2]
        thetas = (
            2 * math.pi * tf.gather(uniform_samples, 1 + even_indices, axis=-1)
        )  # [sample_size, ceil_even(dimensionality)/2]
        cos = tf.cos(thetas)
        sin = tf.sin(thetas)
        samples_tf = tf.reshape(
            tf.stack([Rs * cos, Rs * sin], -1), shape=(sample_size, self._box_muller_req_dim)
        )
        # make sure we only return the number of dimension requested
        samples_tf = samples_tf[:, : self.dimensionality]
        return samples_tf


class QuasiMonteCarloMultivariateNormalSampler:
    """
    Sample from a multivariate d dimensional normal:
    N~(μ, σ)
    """

    def __init__(self, mean: TensorType, cov: TensorType):
        """
        :param mean
        :param cov full covariance matrices
        """

        self._mean = mean
        self._cov = cov
        self._chol_covariance = tf.linalg.cholesky(cov)
        self.base_sampler = QuasiMonteCarloNormalSampler(tf.shape(mean)[-1])

    def sample(self, sample_size: int):
        return self._mean + tf.squeeze(
            tf.matmul(
                self._chol_covariance,
                tf.cast(
                    self.base_sampler.sample(sample_size)[..., tf.newaxis], dtype=self._mean.dtype
                ),
            ),
            -1,
        )
