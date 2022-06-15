# Copyright 2021 The Trieste Contributors
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
This module contains toy objective functions, useful for experimentation. A number of them have been
taken from `this Virtual Library of Simulation Experiments
<https://web.archive.org/web/20211015101644/https://www.sfu.ca/~ssurjano/> (:cite:`ssurjano2021`)`_.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from math import pi
from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp

from ..space import Box
from ..types import TensorType
from ..utils.rmo_utils import estimate_fmean, estimate_fmean_and_fvar, estimate_fvar
from .extra_problems.robot_pushing.push_world import *


class SingleObjectiveTestProblem(ABC):
    """
    Base class for synthetic single-objective test functions.
    Prepares the synthetic function and generates pareto optimal points.
    The latter can be used as a reference of performance measure of certain
    multi-objective optimization algorithms.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        The input dimensionality of the test function
        """

    @property
    @abstractmethod
    def bounds(self) -> list[list[float]]:
        """
        The input space bounds of the test function
        """

    @abstractmethod
    def objective(self) -> Callable[[TensorType], TensorType]:
        """
        Get the synthetic test function.

        :return: A callable synthetic function
        """

    def fmean_objective(self, base_samples: TensorType) -> Callable[[TensorType], TensorType]:
        """
        Get the synthetic test function's fmean w.r.t base distribution
        """

        return partial(estimate_fmean, f=self.objective(), samples=base_samples)

    def fvar_objective(self, base_samples: TensorType) -> Callable[[TensorType], TensorType]:
        """
        Get the synthetic test function's fvar w.r.t base distribution
        """
        return partial(estimate_fvar, f=self.objective(), samples=base_samples)

    def fmean_fvar_objective(self, base_samples: TensorType) -> Callable[[TensorType], TensorType]:
        """
        Get the synthetic test function's fmean, fvar w.r.t base distribution
        """
        return partial(estimate_fmean_and_fvar, f=self.objective(), samples=base_samples)


class Branin(SingleObjectiveTestProblem):
    bounds = [[0.0] * 2, [1.0] * 2]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return branin


def _branin_internals(x: TensorType, scale: TensorType, translate: TensorType) -> TensorType:
    x0 = x[..., :1] * 15.0 - 5.0
    x1 = x[..., 1:] * 15.0

    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)

    return scale * ((x1 - b * x0 ** 2 + c * x0 - r) ** 2 + s * (1 - t) * tf.cos(x0) + translate)


def branin(x: TensorType) -> TensorType:
    """
    The Branin-Hoo function over :math:`[0, 1]^2`. See
    :cite:`Picheny2013` for details.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 2))])

    return _branin_internals(x, 1, 10)


def scaled_branin(x: TensorType) -> TensorType:
    """
    The Branin-Hoo function, rescaled to have zero mean and unit variance over :math:`[0, 1]^2`. See
    :cite:`Picheny2013` for details.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 2))])

    return _branin_internals(x, 1 / 51.95, -44.81)


class SinLinear(SingleObjectiveTestProblem):
    bounds = [[0.0] * 1, [1.0] * 1]
    dim = 1

    def objective(self) -> Callable[[TensorType], TensorType]:
        return synthetic_1d_01


def synthetic_1d_01(x: TensorType) -> TensorType:
    """
    from NMES paper
    """
    f = tf.sin(5 * pi * x ** 2) + 0.5 * x
    return -f


class Bird(SingleObjectiveTestProblem):
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    dim = 2
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization#:~:text=Mishra%27s%20Bird%20function%20%2D%20constrained%5B
    """

    def objective(self) -> Callable[[TensorType], TensorType]:
        return bird_function


def bird_function(at) -> TensorType:
    _real_bounds = tf.constant([[-10.0, -6.5], [0.0, 0.0]], at.dtype)
    at = at * (_real_bounds[-1] - _real_bounds[0]) + _real_bounds[0]
    x = at[..., 0]
    y = at[..., -1]
    return (
        tf.sin(y) * tf.exp((1 - tf.cos(x)) ** 2)
        + tf.cos(x) * tf.exp((1 - tf.sin(y)) ** 2)
        + (x - y) ** 2
    )[..., tf.newaxis]


class Goldstein_Price(SingleObjectiveTestProblem):
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return goldstein_price


def goldstein_price(at) -> TensorType:
    _real_bounds = tf.constant([[-2.0, -2.0], [2.0, 2.0]], at.dtype)
    at = at * (_real_bounds[-1] - _real_bounds[0]) + _real_bounds[0]
    x1 = at[..., 0]
    x2 = at[..., -1]
    return (
        (
            1
            + ((x1 + x2 + 1) ** 2)
            * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
        )
        * (
            30
            + ((2 * x1 - 3 * x2) ** 2)
            * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
        )
    )[..., tf.newaxis]


class Beale(SingleObjectiveTestProblem):
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return beale_function


def beale_function(at) -> TensorType:
    _real_bounds = tf.constant([[-4.5, -4.5], [4.5, 4.5]], at.dtype)
    at = at * (_real_bounds[-1] - _real_bounds[0]) + _real_bounds[0]
    x = at[..., 0]
    y = at[..., -1]
    return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2)[
        ..., tf.newaxis
    ]


class Forrester(SingleObjectiveTestProblem):
    bounds = [[0.0], [1.0]]
    dim = 1

    def objective(self) -> Callable[[TensorType], TensorType]:
        return forrester_function


def forrester_function(at) -> TensorType:
    at = ((6 * at - 2) ** 2) * tf.sin(12 * at - 4)
    return at


_ORIGINAL_BRANIN_MINIMIZERS = tf.constant(
    [[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]], tf.float64
)

BRANIN_MINIMIZERS = (_ORIGINAL_BRANIN_MINIMIZERS + [5.0, 0.0]) / 15.0
"""
The three global minimizers of the :func:`branin` function over :math:`[0, 1]^2`, with shape [3, 2]
and dtype float64.
"""


BRANIN_MINIMUM = tf.constant([0.397887], tf.float64)
""" The global minimum of the :func:`branin` function, with shape [1] and dtype float64. """


class GMM(SingleObjectiveTestProblem):
    bounds = [[0.0] * 2, [1.0] * 2]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return gmm_2d


def gmm_2d(x):
    """
    From NMES
    """
    gmm_pos2 = tf.constant([[[0.2, 0.2]], [[0.8, 0.2]], [[0.5, 0.7]]], dtype=x.dtype)
    gmm_std = tf.constant([[[0.20, 0.20]], [[0.10, 0.10]], [[0.10, 0.10]]], dtype=x.dtype)
    gmm_norm2 = 2 * pi * (gmm_std[..., 0] ** 2) * tf.constant([[0.5], [0.7], [0.7]], dtype=x.dtype)
    gaussians2 = tfp.distributions.MultivariateNormalDiag(loc=gmm_pos2, scale_diag=gmm_std)
    ff = tf.transpose(tf.reduce_sum(gmm_norm2 * gaussians2.prob(x), axis=0, keepdims=True))

    return -ff


# def gmm(x):
#     """
#     Nic's GMM implementation
#     """
#     y = - tf.exp(-0.5 * tf.reduce_sum((x - 1.) ** 2, axis=1)) - \
#         tf.exp(-0.5 * tf.reduce_sum((x + 1.) ** 2, axis=1))
#     return y[:, None]

SCALED_BRANIN_MINIMUM = tf.constant([-1.047393], tf.float64)
""" The global minimum of the :func:`branin` function, with shape [1] and dtype float64. """


BRANIN_SEARCH_SPACE = Box([0.0], [1.0]) ** 2
""" The search space for the :func:`branin` function. """


def simple_quadratic(x: TensorType) -> TensorType:
    """
    A trivial quadratic function over :math:`[0, 1]^2`. Useful for quick testing.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 2))])

    return -tf.math.reduce_sum(x, axis=-1, keepdims=True) ** 2


SIMPLE_QUADRATIC_MINIMIZER = tf.constant([[1.0, 1.0]], tf.float64)
"""
The global minimizer of the :func:`simple_quadratic` function over :math:`[0, 1]^2`,
with shape [1, 2] and dtype float64.
"""

SIMPLE_QUADRATIC_MINIMUM = tf.constant([-4.0], tf.float64)
"""
The global minimum of the :func:`simple_quadratic` function over :math:`[0, 1]^2`, with shape [1]
and dtype float64.
"""

SIMPLE_QUADRATIC_SEARCH_SPACE = BRANIN_SEARCH_SPACE
""" The search space for the :func:`simple_quadratic` function. """


def gramacy_lee(x: TensorType) -> TensorType:
    """
    The Gramacy & Lee function, typically evaluated over :math:`[0.5, 2.5]`. See
    :cite:`gramacy2012cases` for details.

    :param x: Where to evaluate the function, with shape [..., 1].
    :return: The function values, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 1))])
    return tf.sin(10 * math.pi * x) / (2 * x) + (x - 1) ** 4


GRAMACY_LEE_MINIMIZER = tf.constant([[0.548562]], tf.float64)
"""
The global minimizer of the :func:`gramacy_lee` function over :math:`[0.5, 2.5]`, with shape [1, 1]
and dtype float64.
"""


GRAMACY_LEE_MINIMUM = tf.constant([-0.869011], tf.float64)
"""
The global minimum of the :func:`gramacy_lee` function over :math:`[0.5, 2.5]`, with shape [1] and
dtype float64.
"""


GRAMACY_LEE_SEARCH_SPACE = Box([0.5], [2.5])
""" The search space for the :func:`gramacy_lee` function. """


def logarithmic_goldstein_price(x: TensorType) -> TensorType:
    """
    A logarithmic form of the Goldstein-Price function, with zero mean and unit variance over
    :math:`[0, 1]^2`. See :cite:`Picheny2013` for details.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 2))])

    x0, x1 = tf.split(4 * x - 2, 2, axis=-1)

    a = (x0 + x1 + 1) ** 2
    b = 19 - 14 * x0 + 3 * x0 ** 2 - 14 * x1 + 6 * x0 * x1 + 3 * x1 ** 2
    c = (2 * x0 - 3 * x1) ** 2
    d = 18 - 32 * x0 + 12 * x0 ** 2 + 48 * x1 - 36 * x0 * x1 + 27 * x1 ** 2

    return (1 / 2.427) * (tf.math.log((1 + a * b) * (30 + c * d)) - 8.693)


LOGARITHMIC_GOLDSTEIN_PRICE_MINIMIZER = tf.constant([[0.5, 0.25]], tf.float64)
"""
The global minimizer for the :func:`logarithmic_goldstein_price` function, with shape [1, 2] and
dtype float64.
"""


LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM = tf.constant([-3.12913], tf.float64)
"""
The global minimum for the :func:`logarithmic_goldstein_price` function, with shape [1] and dtype
float64.
"""


class SixHumpCamel(SingleObjectiveTestProblem):
    def objective(self) -> Callable[[TensorType], TensorType]:
        return six_hump_camel


def six_hump_camel(x):
    """
    https://www.sfu.ca/~ssurjano/camel6.html
    """
    tf.debugging.assert_shapes([(x, (..., 2))])
    return (
        (4 - 2.1 * x[..., 0] ** 2 + (x[..., 0] ** 4) / 3) * x[..., 0] ** 2
        + x[..., 0] * x[..., 1]
        + (-4 + 4 * x[..., 1] ** 2) * x[..., 1] ** 2
    )


LOGARITHMIC_GOLDSTEIN_PRICE_SEARCH_SPACE = Box([0.0], [1.0]) ** 2
""" The search space for the :func:`logarithmic_goldstein_price` function. """


class Hartmann_3(SingleObjectiveTestProblem):
    bounds = [[0.0] * 3, [1.0] * 3]
    dim = 3

    def objective(self) -> Callable[[TensorType], TensorType]:
        return hartmann_3


def hartmann_3(x: TensorType) -> TensorType:
    """
    The Hartmann 3 test function over :math:`[0, 1]^3`. This function has 3 local
    and one global minima. See https://www.sfu.ca/~ssurjano/hart3.html for details.

    :param x: The points at which to evaluate the function, with shape [..., 3].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 3))])

    a = [1.0, 1.2, 3.0, 3.2]
    A = [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
    P = [
        [0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0381, 0.5743, 0.8828],
    ]

    inner_sum = -tf.reduce_sum(A * (tf.expand_dims(x, -2) - P) ** 2, -1)
    return -tf.reduce_sum(a * tf.math.exp(inner_sum), -1, keepdims=True)


HARTMANN_3_MINIMIZER = tf.constant([[0.114614, 0.555649, 0.852547]], tf.float64)
"""
The global minimizer for the :func:`hartmann_3` function, with shape [1, 3] and
dtype float64.
"""

HARTMANN_3_MINIMUM = tf.constant([-3.86278], tf.float64)
"""
The global minimum for the :func:`hartmann_3` function, with shape [1] and dtype
float64.
"""


HARTMANN_3_SEARCH_SPACE = Box([0.0], [1.0]) ** 3
""" The search space for the :func:`hartmann_3` function. """


def shekel_4(x: TensorType) -> TensorType:
    """
    The Shekel test function over :math:`[0, 1]^4`. This function has ten local
    minima and a single global minimum. See https://www.sfu.ca/~ssurjano/shekel.html for details.
    Note that we rescale the original problem, which is typically defined
    over `[0, 10]^4`.

    :param x: The points at which to evaluate the function, with shape [..., 4].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 4))])

    y: TensorType = x * 10.0

    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    C = [
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
    ]

    inner_sum = tf.reduce_sum((tf.expand_dims(y, -1) - C) ** 2, 1)
    inner_sum += tf.cast(tf.transpose(beta), dtype=inner_sum.dtype)
    return -tf.reduce_sum(inner_sum ** (-1), -1, keepdims=True)


SHEKEL_4_MINIMIZER = tf.constant([[0.4, 0.4, 0.4, 0.4]], tf.float64)
"""
The global minimizer for the :func:`shekel_4` function, with shape [1, 4] and
dtype float64.
"""

SHEKEL_4_MINIMUM = tf.constant([-10.5363], tf.float64)
"""
The global minimum for the :func:`shekel_4` function, with shape [1] and dtype
float64.
"""


SHEKEL_4_SEARCH_SPACE = Box([0.0], [1.0]) ** 4
""" The search space for the :func:`shekel_4` function. """


def rosenbrock_4(x: TensorType) -> TensorType:
    """
    The Rosenbrock function, rescaled to have zero mean and unit variance over :math:`[0, 1]^4. See
    :cite:`Picheny2013` for details.
    This function (also known as the Banana function) is unimodal, however the minima
    lies in a narrow valley.

    :param x: The points at which to evaluate the function, with shape [..., 4].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 4))])

    y: TensorType = x * 15.0 - 5
    unscaled_function = tf.reduce_sum(
        (100.0 * (y[..., 1:] - y[..., :-1]) ** 2 + (1 - y[..., :-1]) ** 2), axis=-1, keepdims=True
    )
    return (unscaled_function - 3.827 * 1e5) / (3.755 * 1e5)


ROSENBROCK_4_MINIMIZER = tf.constant([[0.4, 0.4, 0.4, 0.4]], tf.float64)
"""
The global minimizer for the :func:`rosenbrock_4` function, with shape [1, 4] and
dtype float64.
"""

ROSENBROCK_4_MINIMUM = tf.constant([-1.01917], tf.float64)
"""
The global minimum for the :func:`rosenbrock_4` function, with shape [1] and dtype
float64.
"""


ROSENBROCK_4_SEARCH_SPACE = Box([0.0], [1.0]) ** 4
""" The search space for the :func:`rosenbrock_4` function. """


def ackley_5(x: TensorType) -> TensorType:
    """
    The Ackley test function over :math:`[0, 1]^5`. This function has
    many local minima and a global minima. See https://www.sfu.ca/~ssurjano/ackley.html
    for details.
    Note that we rescale the original problem, which is typically defined
    over `[-32.768, 32.768]`.

    :param x: The points at which to evaluate the function, with shape [..., 5].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 5))])

    x = (x - 0.5) * (32.768 * 2.0)

    exponent_1 = -0.2 * tf.math.sqrt((1 / 5.0) * tf.reduce_sum(x ** 2, -1))
    exponent_2 = (1 / 5.0) * tf.reduce_sum(tf.math.cos(2.0 * math.pi * x), -1)

    function = (
        -20.0 * tf.math.exp(exponent_1)
        - tf.math.exp(exponent_2)
        + 20.0
        + tf.cast(tf.math.exp(1.0), dtype=tf.float64)
    )

    return tf.expand_dims(function, -1)


ACKLEY_5_MINIMIZER = tf.constant([[0.5, 0.5, 0.5, 0.5, 0.5]], tf.float64)
"""
The global minimizer for the :func:`ackley_5` function, with shape [1, 5] and
dtype float64.
"""

ACKLEY_5_MINIMUM = tf.constant([0.0], tf.float64)
"""
The global minimum for the :func:`ackley_5` function, with shape [1] and dtype
float64.
"""

ACKLEY_5_SEARCH_SPACE = Box([0.0], [1.0]) ** 5
""" The search space for the :func:`ackley_5` function. """


class Hartmann_6(SingleObjectiveTestProblem):
    bounds = [[0.0] * 6, [1.0] * 6]
    dim = 6

    def objective(self) -> Callable[[TensorType], TensorType]:
        return hartmann_6


def hartmann_6(x: TensorType) -> TensorType:
    """
    The Hartmann 6 test function over :math:`[0, 1]^6`. This function has
    6 local and one global minima. See https://www.sfu.ca/~ssurjano/hart6.html
    for details.

    :param x: The points at which to evaluate the function, with shape [..., 6].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 6))])

    a = [1.0, 1.2, 3.0, 3.2]
    A = [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
    P = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]

    inner_sum = -tf.reduce_sum(A * (tf.expand_dims(x, 1) - P) ** 2, -1)
    return -tf.reduce_sum(a * tf.math.exp(inner_sum), -1, keepdims=True)


HARTMANN_6_MINIMIZER = tf.constant(
    [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]], tf.float64
)
"""
The global minimizer for the :func:`hartmann_6` function, with shape [1, 6] and
dtype float64.
"""

HARTMANN_6_MINIMUM = tf.constant([-3.32237], tf.float64)
"""
The global minimum for the :func:`hartmann_6` function, with shape [1] and dtype
float64.
"""


HARTMANN_6_SEARCH_SPACE = Box([0.0], [1.0]) ** 6
""" The search space for the :func:`hartmann_6` function. """


def michalewicz(x: TensorType, d: int = 2, m: int = 10) -> TensorType:
    """
    The Michalewicz function over :math:`[0, \\pi]` for all i=1,...,d. Dimensionality is determined
    by the parameter ``d`` and it features steep ridges and drops. It has :math:`d!` local minima,
    and it is multimodal. The parameter ``m`` defines the steepness of they valleys and ridges; a
    larger ``m`` leads to a more difficult search. The recommended value of ``m`` is 10. See
    https://www.sfu.ca/~ssurjano/egg.html for details.
    :param x: The points at which to evaluate the function, with shape [..., d].
    :param d: The dimension of the function.
    :param m: The steepness of the valleys/ridges.
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_greater_equal(d, 1)
    tf.debugging.assert_shapes([(x, (..., d))])

    xi = tf.range(1, (d + 1), delta=1, dtype=tf.float64) * tf.pow(x, 2)
    result = tf.reduce_sum(tf.sin(x) * tf.pow(tf.sin(xi / math.pi), 2 * m), axis=1, keepdims=True)

    return -result


def michalewicz_2(x: TensorType) -> TensorType:
    """
    Convenience function for the 2-dimensional :func:`michalewicz` function with steepness 10.
    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    """
    return michalewicz(x, d=2)


def michalewicz_5(x: TensorType) -> TensorType:
    """
    Convenience function for the 5-dimensional :func:`michalewicz` function with steepness 10.
    :param x: The points at which to evaluate the function, with shape [..., 5].
    :return: The function values at ``x``, with shape [..., 1].
    """
    return michalewicz(x, d=5)


def michalewicz_10(x: TensorType) -> TensorType:
    """
    Convenience function for the 10-dimensional :func:`michalewicz` function with steepness 10.
    :param x: The points at which to evaluate the function, with shape [..., 10].
    :return: The function values at ``x``, with shape [..., 1].
    """
    return michalewicz(x, d=10)


MICHALEWICZ_2_MINIMIZER = tf.constant([[2.202906, 1.570796]], tf.float64)
"""
The global minimizer of the :func:`michalewicz` function over :math:`[0, \\pi]^2`,
with shape [1, 2] and dtype float64. Taken from https://arxiv.org/abs/2003.09867
"""


MICHALEWICZ_5_MINIMIZER = tf.constant(
    [[2.202906, 1.570796, 1.284992, 1.923058, 1.720470]], tf.float64
)
"""
The global minimizer of the :func:`michalewicz` function over :math:`[0, \\pi]^5`,
with shape [1, 5] and dtype float64. Taken from https://arxiv.org/abs/2003.09867
"""


MICHALEWICZ_10_MINIMIZER = tf.constant(
    [
        [
            2.202906,
            1.570796,
            1.284992,
            1.923058,
            1.720470,
            1.570796,
            1.454414,
            1.756087,
            1.655717,
            1.570796,
        ]
    ],
    tf.float64,
)
"""
The global minimizer of the :func:`michalewicz` function over :math:`[0, \\pi]^10`,
with shape [1, 10] and dtype float64. Taken from https://arxiv.org/abs/2003.09867
"""


MICHALEWICZ_2_MINIMUM = tf.constant([-1.8013034], tf.float64)
"""
The global minimum of the 2-dimensional :func:`michalewicz` function, with shape [1] and dtype
float64. Taken from https://arxiv.org/abs/2003.09867
"""


MICHALEWICZ_5_MINIMUM = tf.constant([-4.6876582], tf.float64)
"""
The global minimum of the 5-dimensional :func:`michalewicz` function, with shape [1] and dtype
float64. Taken from https://arxiv.org/abs/2003.09867
"""


MICHALEWICZ_10_MINIMUM = tf.constant([-9.6601517], tf.float64)
"""
The global minimum of the 10-dimensional :func:`michalewicz` function, with shape [1] and dtype
float64. Taken from https://arxiv.org/abs/2003.09867
"""


MICHALEWICZ_2_SEARCH_SPACE = Box([0.0], [pi]) ** 2
"""
The search space for the 2-dimensional :func:`michalewicz` function.
"""


MICHALEWICZ_5_SEARCH_SPACE = Box([0.0], [pi]) ** 5
"""
The search space for the 5-dimensional :func:`michalewicz` function.
"""


MICHALEWICZ_10_SEARCH_SPACE = Box([0.0], [pi]) ** 10
"""
The search space for the 10-dimensional :func:`michalewicz` function.
"""


def trid(x: TensorType, d: int = 10) -> TensorType:
    """
    The Trid function over :math:`[-d^2, d^2]` for all i=1,...,d. Dimensionality is determined
    by the parameter ``d`` and it has a global minimum. This function has large variation in
    output which makes it challenging for Bayesian optimisation with vanilla Gaussian processes
    with non-stationary kernels. Models that can deal with non-stationarities, such as deep
    Gaussian processes, can be useful for modelling these functions. See :cite:`hebbal2019bayesian`
    and https://www.sfu.ca/~ssurjano/trid.html for details.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_greater_equal(d, 2)
    tf.debugging.assert_shapes([(x, (..., d))])

    result = tf.reduce_sum(tf.pow(x - 1, 2), 1, True) - tf.reduce_sum(x[:, 1:] * x[:, :-1], 1, True)
    return result


def trid_10(x: TensorType) -> TensorType:
    return trid(x, d=10)


TRID_10_MINIMIZER = tf.constant([[i * (10 + 1 - i) for i in range(1, 10 + 1)]], tf.float64)
"""
The global minimizer of :func:`trid` function is defined as :math:`x_i=i(d+1-i)` for all i=1,...,d.
Here, we define it specifically for the 10-dimensional variant, with shape [1, 10] and dtype
float64.
"""


TRID_10_MINIMUM = tf.constant([-10 * (10 + 4) * (10 - 1) / 6], tf.float64)
"""
The global minimum of :func:`trid` function is defined as :math:`d(d+4)(d-1)/6` for dimensionality
`d`. Here, we define it specifically for the 10-dimensional variant, with shape [1] and dtype
float64.
"""


TRID_10_SEARCH_SPACE = Box([-(10 ** 2)], [10 ** 2]) ** 10
"""
The search space for :func:`trid` function is defined over :math:`[-d^2, d^2]` for all i=1,...,d.
Here, we define it specifically for the 10-dimensional variant.
"""


class Conceptual_Low_Drag_Wing_Design(SingleObjectiveTestProblem):
    bounds = [[0.0] * 2, [1.0] * 2]
    dim = 2

    def __init__(self, weight: float, velocity: float):
        # Note!!! weight is deprecated and not useful
        self._weight = weight
        self._velocity = velocity

    def objective(self) -> Callable[[TensorType], TensorType]:
        return partial(drag_coefficient_calculation, weight = self._weight, velocity = self._velocity)


lb_wing = tf.constant([[1.0, 0.05]])
ub_wing = tf.constant([[3.0, 1.0]])


def drag_coefficient_calculation(ats, weight, velocity):
    ats = ats * tf.cast((ub_wing - lb_wing), ats.dtype) + tf.cast(lb_wing, ats.dtype)
    _rho = tf.constant([1.29], dtype=ats.dtype)
    _e = 0.96
    _N_ult = 2.5
    W0 = 18.0
    _mu = 17.8 * 1e-6
    _k = 1.2
    from scipy.optimize import fsolve
    if tf.rank(ats) == 3:
        spanned_ats = tf.reshape(ats, (tf.shape(ats)[0] * tf.shape(ats)[1], tf.shape(ats)[-1]))
    else:
        spanned_ats = ats
    spans, chords = tf.split(spanned_ats, 2, -1)
    res = []
    for span, chord in zip(spans, chords):
        S = span * chord

        def W_func(W):
            return W0 + 45.42 * S + 8.71 * 1e-5 \
                   * (_N_ult * (span ** 3) * tf.sqrt(W0 * tf.cast(W, dtype=S.dtype))/(S * 0.12)) - (W-W0)
        w_guess = tf.constant([40.0], dtype=ats.dtype)
        W_solution = fsolve(W_func, np.asarray(w_guess))[0]
        Re = _rho * velocity * chord / _mu

        def calc_Cf(raynould_number):
            return 0.074 / raynould_number ** 0.2
        C_f =calc_Cf(Re)
        CL = 2 * W_solution / (_rho * velocity ** 2 * S)
        A = span / chord
        CD = 0.03062702/S + _k * C_f * 2.05 + CL ** 2 / (math.pi * A * _e)
        res.append(tf.math.log(-(5 * CL / CD - (tf.maximum(A, 10) - 10) -
                                 100 * (tf.maximum(CL - 0.9, 0.0)) - 0.4 * W_solution - 5)))
    return_res = tf.convert_to_tensor(res, dtype=ats.dtype)
    if tf.rank(ats) == 3:
        return_res = tf.reshape(return_res, (*tf.shape(ats)[:-1], 1))
    return return_res


class Robot_Pushing_2D(SingleObjectiveTestProblem):
    bounds = [[0.0] * 3, [1.0] * 3]
    dim = 2

    def __init__(self, obj_loc_x: float, obj_loc_y: float, pushing_time: float):
        self._obj_loc_x = obj_loc_x
        self._obj_loc_y = obj_loc_y
        self._pushing_time = pushing_time

    def objective(self) -> Callable[[TensorType], TensorType]:
        return partial(
            robot_pushing_2d, gx=self._obj_loc_x, gy=self._obj_loc_y, t=self._pushing_time
        )


class Robot_Pushing_3D(SingleObjectiveTestProblem):
    bounds = [[0.0] * 3, [1.0] * 3]
    dim = 3

    def __init__(self, obj_loc_x: float, obj_loc_y: float, seed: int = None):
        if seed is not None:
            # We use numpy instead of tf is because np is fully reproducible
            import numpy as np

            np.random.seed(seed)
            obj_loc_x, obj_loc_y = np.random.uniform([1.0, 1.0], [5.0, 5.0], 2)
            # print('X', obj_loc_x)
            # print('Y', obj_loc_y)
        self._obj_loc_x = obj_loc_x
        self._obj_loc_y = obj_loc_y

    def objective(self) -> Callable[[TensorType], TensorType]:
        return partial(robot_pushing_3d, gx=self._obj_loc_x, gy=self._obj_loc_y)


lb_rp2d = tf.constant([[0, 0]])
ub_rp2d = tf.constant([[5, 5]])


def robot_pushing_2d(xs: TensorType, t, gx, gy) -> TensorType:
    """
    The Hartmann 6 test function over :math:`[0, 1]^6`. This function has
    6 local and one global minima. See https://www.sfu.ca/~ssurjano/hart6.html
    for details.

    :param x: The points at which to evaluate the function, with shape [..., 6].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.

    rx = float(sys.argv[1])
    ry = float(sys.argv[2])
    gx = float(sys.argv[4])
    gy = float(sys.argv[5])
    simu_steps = int(float(sys.argv[3]) * 10)

    """
    tf.debugging.assert_shapes([(xs, (..., 2))])
    print("eval")
    xs = xs * (tf.cast(ub_rp2d, dtype=xs.dtype) - tf.cast(lb_rp2d, dtype=xs.dtype)) + tf.cast(
        lb_rp2d, dtype=xs.dtype
    )
    res = []
    for x in xs:
        rx, ry = x.numpy()
        rx = float(rx)
        ry = float(ry)
        res.append(np.atleast_1d(_problem(rx, ry, t, gx, gy)))
    return tf.convert_to_tensor(res, dtype=xs.dtype)


lb_rp3d = tf.constant([[-5, -5, 1]])
ub_rp3d = tf.constant([[5, 5, 30]])


def robot_pushing_3d(xs: TensorType, gx, gy) -> TensorType:
    """
    The Hartmann 6 test function over :math:`[0, 1]^6`. This function has
    6 local and one global minima. See https://www.sfu.ca/~ssurjano/hart6.html
    for details.

    :param x: The points at which to evaluate the function, with shape [..., 6].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.

    rx = float(sys.argv[1])
    ry = float(sys.argv[2])
    gx = float(sys.argv[4])
    gy = float(sys.argv[5])
    simu_steps = int(float(sys.argv[3]) * 10)

    """
    tf.debugging.assert_shapes([(xs, (..., 3))])
    xs = xs * (tf.cast(ub_rp3d, dtype=xs.dtype) - tf.cast(lb_rp3d, dtype=xs.dtype)) + tf.cast(
        lb_rp3d, dtype=xs.dtype
    )
    res = []
    for x in xs.numpy():
        rx, ry, t = x
        rx = float(rx)
        ry = float(ry)
        res.append(np.atleast_1d(_problem(rx, ry, t, gx, gy)))
    return tf.convert_to_tensor(res, dtype=xs.dtype)


def _problem(rx, ry, t, gx, gy):
    simu_steps = int(t * 10)
    world = b2WorldInterface(False)
    oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = (
        "circle",
        1,
        0.01,
        0.05,
        0.01,
        "rectangle",
        (0.3, 1),
    )
    thing, base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0, 0))

    rx = 1e-5 if np.abs(rx) < 1e-5 else rx
    init_angle = np.arctan(ry / rx)
    robot = end_effector(world, (rx, ry), base, init_angle, hand_shape, hand_size)
    ret = simu_push(world, thing, robot, base, simu_steps)
    ret = np.linalg.norm(np.array([gx, gy]) - ret)
    return ret
