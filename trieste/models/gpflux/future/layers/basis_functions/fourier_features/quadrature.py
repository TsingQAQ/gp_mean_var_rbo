#
# Copyright (c) 2021 The GPflux Contributors.
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
#
""" A kernel's features and coefficients using quadrature Fourier features (QFF). """
import warnings
from typing import Mapping

import gpflow
import tensorflow as tf
from gpflow.base import TensorType
from gpflow.quadrature.gauss_hermite import ndgh_points_and_weights
from gpflux.types import ShapeType

from ...basis_functions.fourier_features.base import FourierFeaturesBase
from ...basis_functions.fourier_features.utils import QFF_SUPPORTED_KERNELS, _bases_concat


class QuadratureFourierFeatures(FourierFeaturesBase):
    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):
        """
        Args:
            kernel:
            **kwargs:
        """
        assert isinstance(kernel, QFF_SUPPORTED_KERNELS), "Unsupported Kernel"
        if tf.reduce_any(tf.less(kernel.lengthscales, 1e-1)):
            warnings.warn(
                "Quadrature Fourier feature approximation of kernels "
                "with small lengthscale lead to unexpected behaviors!"
            )
        super(QuadratureFourierFeatures, self).__init__(kernel, n_components, **kwargs)

    def build(self, input_shape: ShapeType) -> None:
        """
        Creates the variables of the layer.
        See `tf.keras.layers.Layer.build()
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#build>`_.
        """
        input_dim = input_shape[-1]
        abscissa_value, omegas_value = ndgh_points_and_weights(
            dim=input_dim, n_gh=self.n_components
        )  # [N1 * N2 * ..., * Nd, d], [N1 * N2 * ..., * Nd, 1]
        omegas_value = tf.squeeze(omegas_value, axis=-1)

        # Quadrature node points
        self.abscissa = tf.Variable(initial_value=abscissa_value, trainable=False)  # (M^D, D)
        # Gauss-Hermite weights
        self.factors = tf.Variable(initial_value=omegas_value, trainable=False)  # (M^D,)
        super(QuadratureFourierFeatures, self).build(input_shape)

    def _compute_output_dim(self, input_shape: ShapeType) -> int:
        """
        Args:
            input_shape: input dimensionality
        Returns:

        """
        input_dim = input_shape[-1]
        return 2 * self.n_components ** input_dim

    def _compute_constant(self) -> tf.Tensor:
        """
        Returns:
        """
        return tf.tile(tf.sqrt(self.kernel.variance * self.factors), multiples=[2])  # [2m]

    def _compute_bases(self, inputs: TensorType) -> tf.Tensor:
        """
        :param inputs: [N, D]
        :return: [N, 2M^D]
        """
        return _bases_concat(
            inputs, self.abscissa
        )  # [N, D], [M^D, D] -> tf.concat([N, M], [N, M]) -> [N, 2M^D]
