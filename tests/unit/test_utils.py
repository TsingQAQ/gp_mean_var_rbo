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
from typing import Callable, Mapping

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from trieste.type import TensorType
from trieste.utils import K, U, V, map_values
from trieste.utils.multi_objectives import DTLZ1, VLMOP2
from trieste.utils.rmo_utils import get_query_points_fmean


@pytest.mark.parametrize(
    "f, mapping, expected",
    [
        (abs, {}, {}),
        (abs, {1: -1, -2: 2}, {1: 1, -2: 2}),
        (len, {"a": [1, 2, 3], "b": [4, 5]}, {"a": 3, "b": 2}),
    ],
)
def test_map_values(f: Callable[[U], V], mapping: Mapping[K, U], expected: Mapping[K, V]) -> None:
    assert map_values(f, mapping) == expected


@pytest.mark.parametrize(
    "query_points, f, noise_seed_samples, noise_scale, noise_type",
    [
        (
            tf.constant([[0.3, 0.2], [0.7, 0.5], [0.4, 0.8]]),
            VLMOP2().objective(),
            tfp.distributions.Normal(loc=[0.0, 0.0], scale=[0.03, 0.01]).sample(5000),
            tf.constant([0.03, 0.01]),
            "normal",
        ),
        (
            tf.constant([[0.3, 0.2], [0.7, 0.5], [0.4, 0.8]]),
            DTLZ1().objective(),
            tfp.distributions.Uniform(-tf.constant([0.2, 0.05]), tf.constant([0.2, 0.05])).sample(
                5000
            ),
            tf.constant([0.2, 0.05]),
            "uniform",
        ),
    ],
)
def test_get_fmean_dataset_with_input(
    query_points: TensorType,
    f: Callable,
    noise_seed_samples: TensorType,
    noise_scale: TensorType,
    noise_type: str,
) -> None:
    if noise_type == "normal":
        tf.debugging.assert_near(
            get_query_points_fmean(query_points, f, noise_seed_samples),
            _get_real_fmean_dataset_normal(
                query_points, f, noise_scale, x_mc_num=noise_seed_samples.shape[0]
            ),
            rtol=0.005,
        )
    elif noise_type == "uniform":
        tf.debugging.assert_near(
            get_query_points_fmean(query_points, f, noise_seed_samples),
            _get_real_fmean_dataset_uniform(
                query_points, f, noise_scale, x_mc_num=noise_seed_samples.shape[0]
            ),
            rtol=0.02,
        )
    else:
        raise NotImplementedError("Noise Type not understood")


def _get_real_fmean_dataset_normal(
    query_points: TensorType, f: Callable, noise_scales: TensorType, x_mc_num=10000
):
    """
    :param query_points
    :param f
    :param noise_scales
    :param x_mc_num
    """
    X = query_points
    tf.debugging.assert_shapes([(noise_scales, ["dim"])])
    # [x_mc_num, batch_size, x_dim]
    Xs = tfp.distributions.Normal(
        X, scale=tf.cast(tf.tile(tf.expand_dims(noise_scales, 0), [X.shape[0], 1]), dtype=X.dtype)
    ).sample(x_mc_num)
    Xs = tf.reshape(Xs, shape=(X.shape[0] * x_mc_num, X.shape[1]))  # [x_mc_num * batch_size, x_dim]
    n_fs = f(Xs)
    n_fs = tf.reshape(n_fs, shape=(x_mc_num, X.shape[0], n_fs.shape[-1]))
    Fs_mean = tf.reduce_mean(n_fs, axis=0)
    return Fs_mean


def _get_real_fmean_dataset_uniform(
    query_points: TensorType, f: Callable, noise_scales: TensorType, x_mc_num=10000
):
    """
    :param query_points
    :param f
    :param noise_scales
    :param x_mc_num
    """
    X = query_points
    tf.debugging.assert_shapes([(noise_scales, ["dim"])])
    Xs = tfp.distributions.Uniform(X - noise_scales, X + noise_scales).sample(x_mc_num)
    # [x_mc_num, batch_size, x_dim]
    Xs = tf.reshape(Xs, shape=(X.shape[0] * x_mc_num, X.shape[1]))  # [x_mc_num * batch_size, x_dim]
    n_fs = f(Xs)
    n_fs = tf.reshape(n_fs, shape=(x_mc_num, X.shape[0], n_fs.shape[-1]))
    Fs_mean = tf.reduce_mean(n_fs, axis=0)
    return Fs_mean
