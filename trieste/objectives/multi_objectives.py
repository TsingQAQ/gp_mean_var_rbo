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
This module contains synthetic multi-objective functions, useful for experimentation.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from math import pi

import tensorflow as tf
import tensorflow_probability as tfp

from ..types import TensorType
from ..utils.rmo_utils import estimate_fmean


class MultiObjectiveTestProblem(ABC):
    """
    Base class for synthetic multi-objective test functions.
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

    @abstractmethod
    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        """
        Generate `n` Pareto optimal points.

        :param n: The number of pareto optimal points to be generated.
        :param seed: An integer used to create a random seed for distributions that
         used to generate pareto optimal points.
        :return: The Pareto optimal points
        """
        pass

    def fmean_objective(self, base_samples: TensorType) -> Callable[[TensorType], TensorType]:
        """
        Get the synthetic test function's fmean w.r.t base distribution

        :return: fmean objective function
        """
        return partial(estimate_fmean, f=self.objective(), samples=base_samples)


class VLMOP2(MultiObjectiveTestProblem):
    """
    The VLMOP2 problem, typically evaluated over :math:`[-2, 2]^2`.
    The idea pareto fronts lies on -1/sqrt(2) - 1/sqrt(2) and x1=x2.
    See :cite:`van1999multiobjective` and  :cite: fonseca1995multiobjective
    (the latter for discussion of pareto front property) for details.
    """

    reference_point = tf.constant([1.0, 1.0])
    bounds = [[-2.0] * 2, [2.0] * 2]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return vlmop2

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        tf.debugging.assert_greater(n, 0)
        _x = tf.linspace([-1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0)], n)
        return vlmop2(tf.concat([_x, _x], axis=1))


def vlmop2(x: TensorType) -> TensorType:
    """
    The VLMOP2 synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 2].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 2))], message="vlmop2 only allow 2d input")
    transl = 1 / math.sqrt(2.0)
    y1 = 1 - tf.exp(-1 * tf.reduce_sum((x - transl) ** 2, axis=-1))
    y2 = 1 - tf.exp(-1 * tf.reduce_sum((x + transl) ** 2, axis=-1))
    return tf.stack([y1, y2], axis=-1)


class DTLZ(MultiObjectiveTestProblem):
    """
    DTLZ series multi-objective test problem.
    See :cite:deb2002scalable for details.
    """

    def __init__(self, input_dim: int, num_objective: int):
        tf.debugging.assert_greater(input_dim, 0)
        tf.debugging.assert_greater(num_objective, 0)
        tf.debugging.assert_greater(
            input_dim,
            num_objective,
            f"input dimension {input_dim}"
            f"  must be greater than function objective numbers {num_objective}",
        )
        self._dim = input_dim
        self.M = num_objective
        self.k = self._dim - self.M + 1
        self._bounds = [[0.0] * input_dim, [1.0] * input_dim]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> list[list[float]]:
        return self._bounds


class DTLZ1(DTLZ):
    """
    The DTLZ1 problem, the idea pareto fronts lie on a linear hyper-plane.
    See :cite:deb2002scalable for details.
    """

    def objective(self) -> Callable[[TensorType], TensorType]:
        return partial(dtlz1, m=self.M, k=self.k, d=self.dim)

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        tf.debugging.assert_greater_equal(self.M, 2)
        rnd = tf.random.uniform([n, self.M - 1], minval=0, maxval=1, seed=seed)
        strnd = tf.sort(rnd, axis=-1)
        strnd = tf.concat([tf.zeros([n, 1]), strnd, tf.ones([n, 1])], axis=-1)
        return 0.5 * (strnd[..., 1:] - strnd[..., :-1])


def dtlz1(x: TensorType, m: int, k: int, d: int) -> TensorType:
    """
    The DTLZ1 synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param m: The objective numbers.
    :param k: The input dimensionality for g.
    :param d: The dimensionality of the synthetic function.
    :return: The function values at ``x``, with shape [..., m].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes(
        [(x, (..., d))],
        message=f"input x dim: {x.shape[-1]} is not align with pre-specified dim: {d}",
    )
    tf.debugging.assert_greater(m, 0, message=f"positive objective numbers expected but found {m}")

    def g(xM: TensorType) -> TensorType:
        return 100 * (
            k
            + tf.reduce_sum(
                (xM - 0.5) ** 2 - tf.cos(20 * math.pi * (xM - 0.5)), axis=-1, keepdims=True
            )
        )

    ta = tf.TensorArray(x.dtype, size=m)
    for i in range(m):
        xM = x[..., m - 1 :]
        y = 1 + g(xM)
        y *= 1 / 2 * tf.reduce_prod(x[..., : m - 1 - i], axis=-1, keepdims=True)
        if i > 0:
            y *= 1 - x[..., m - i - 1, tf.newaxis]
        ta = ta.write(i, y)

    return tf.squeeze(tf.concat(tf.split(ta.stack(), m, axis=0), axis=-1), axis=0)


class DTLZ2(DTLZ):
    """
    The DTLZ2 problem, the idea pareto fronts lie on (part of) a unit hyper sphere.
    See :cite:deb2002scalable for details.
    """

    def objective(self) -> Callable[[TensorType], TensorType]:
        return partial(dtlz2, m=self.M, d=self.dim)

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        tf.debugging.assert_greater_equal(self.M, 2)
        rnd = tf.random.normal([n, self.M], seed=seed)
        samples = tf.abs(rnd / tf.norm(rnd, axis=-1, keepdims=True))
        return samples


def dtlz2(x: TensorType, m: int, d: int) -> TensorType:
    """
    The DTLZ2 synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param m: The objective numbers.
    :param d: The dimensionality of the synthetic function.
    :return: The function values at ``x``, with shape [..., m].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes(
        [(x, (..., d))],
        message=f"input x dim: {x.shape[-1]} is not align with pre-specified dim: {d}",
    )
    tf.debugging.assert_greater(m, 0, message=f"positive objective numbers expected but found {m}")

    def g(xM: TensorType) -> TensorType:
        z = (xM - 0.5) ** 2
        return tf.reduce_sum(z, axis=-1, keepdims=True)

    ta = tf.TensorArray(x.dtype, size=m)
    for i in tf.range(m):
        y = 1 + g(x[..., m - 1 :])
        for j in tf.range(m - 1 - i):
            y *= tf.cos(math.pi / 2 * x[..., j, tf.newaxis])
        if i > 0:
            y *= tf.sin(math.pi / 2 * x[..., m - 1 - i, tf.newaxis])
        ta = ta.write(i, y)

    return tf.squeeze(tf.concat(tf.split(ta.stack(), m, axis=0), axis=-1), axis=0)


class Four_Bar_Truss(MultiObjectiveTestProblem):
    bounds = [[0] * 4, [1] * 4]
    dim = 4

    def objective(self) -> Callable[[TensorType], TensorType]:
        return fourbartruss


def fourbartruss(x) -> TensorType:
    """
    refer: https://github.com/ryojitanabe/reproblems/blob/master/doc/re-supplementary_file.pdf
    P.238 of https://arc.aiaa.org/doi/abs/10.2514/4.866234
    https://www.researchgate.net/publication/234868085_Pareto-Optimal_Solutions_for_a_Truss_Problem
    """
    L = 200  # 200 cm
    F = 10  # 10 KN
    E = 2e5  # KN/cm^2
    sigma = 10  # KN/cm^2
    alpha = F / sigma
    lb = tf.constant([1, math.sqrt(2.0), math.sqrt(2.0), 1], dtype=x.dtype) * alpha
    ub = tf.constant([3] * 4, dtype=x.dtype) * alpha
    x = x * (ub - lb) + lb
    obj1 = L * (
        2 * x[..., 0]
        + tf.cast(tf.sqrt(2.0), dtype=x.dtype) * x[..., 1]
        + tf.sqrt(x[..., 2])
        + x[..., 3]
    )
    obj2 = (
        F
        * L
        / E
        * (
            2.0 / x[..., 0]
            + 2.0 * tf.cast(tf.sqrt(2.0), dtype=x.dtype) / x[..., 1]
            - 2.0 * tf.cast(tf.sqrt(2.0), dtype=x.dtype) / x[..., 2]
            + 2.0 / x[..., 3]
        )
    )
    return tf.stack([obj1, obj2], axis=-1)


class VehicleCrashSafty(MultiObjectiveTestProblem):
    bounds = [[0] * 5, [1] * 5]
    dim = 5

    def objective(self) -> Callable[[TensorType], TensorType]:
        return vehicle_crash_safety_problem


def vehicle_crash_safety_problem(x) -> TensorType:
    lb = tf.constant([1] * 5, dtype=x.dtype)
    ub = tf.constant([3] * 5, dtype=x.dtype)
    x = x * (ub - lb) + lb
    f1 = (
        1640.2823
        + 2.3573285 * x[..., 0]
        + 2.3220035 * x[..., 1]
        + 4.5688768 * x[..., 2]
        + 7.7213633 * x[..., 3]
        + 4.4559504 * x[..., 4]
    )
    f2 = (
        6.5856
        + 1.15 * x[..., 0]
        - 1.0427 * x[..., 1]
        + 0.9738 * x[..., 2]
        + 0.8364 * x[..., 3]
        - 0.3695 * x[..., 0] * x[..., 3]
        + 0.0861 * x[..., 0] * x[..., 4]
        + 0.3628 * x[..., 1] * x[..., 3]
        - 0.1106 * x[..., 1] ** 2
        - 0.3437 * x[..., 2] ** 2
        + 0.1764 * x[..., 3] ** 2
    )
    f3 = (
        -0.0551
        + 0.0181 * x[..., 0]
        + 0.1024 * x[..., 1]
        + 0.0421 * x[..., 2]
        - 0.0073 * x[..., 0] * x[..., 1]
        + 0.024 * x[..., 1] * x[..., 2]
        - 0.0118 * x[..., 1] * x[..., 3]
        - 0.0204 * x[..., 2] * x[..., 3]
        - 0.008 * x[..., 2] * x[..., 4]
        - 0.0241 * x[..., 1] ** 2
        + 0.0109 * x[..., 3] ** 2
    )
    f_X = tf.stack([f1, f2, f3], axis=-1)
    return f_X


class The_Triangle_Problem(MultiObjectiveTestProblem):
    bounds = [[0] * 2, [1] * 2]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return the_triangle_problem


def the_triangle_problem(x) -> TensorType:
    """
    refer:
    @incollection{rivier2018surrogate,
    title={Surrogate-Assisted Bounding-Box Approach for Optimization Problems with Approximated Objectives},
    author={Rivier, Mickael and Congedo, Pietro Marco},
    booktitle={Research Report RR-9155},
    year={2018},
    publisher={Inria}
    }
    """
    lb = tf.constant([1.0, 1.0], dtype=x.dtype)
    ub = tf.constant([10] * 2, dtype=x.dtype)
    x = x * (ub - lb) + lb
    obj1 = (x[..., 0] + x[..., 1]) / 10 + tf.abs(x[..., 0] - x[..., 1])
    obj2 = (x[..., 0]) / 5 + tf.abs(x[..., 1] - 2)
    return tf.stack([obj1, obj2], axis=-1)


class BraninCurrin(MultiObjectiveTestProblem):
    bounds = [[0, 0], [1, 1]]
    dim = 2

    def objective(self):
        return branincurrin


def _branin(X: tf.Tensor) -> tf.Tensor:
    t1 = (
        X[..., 1, tf.newaxis]
        - 5.1 / (4 * pi ** 2) * X[..., 0, tf.newaxis] ** 2
        + 5 / pi * X[..., 0, tf.newaxis]
        - 6
    )
    t2 = 10 * (1 - 1 / (8 * pi)) * tf.cos(X[..., 0, tf.newaxis])
    return t1 ** 2 + t2 + 10


def _rescaled_branin(X: tf.Tensor) -> tf.Tensor:
    # return to Branin bounds
    x_0 = 15 * X[..., 0, tf.newaxis] - 5
    x_1 = 15 * X[..., 1, tf.newaxis]
    return _branin(tf.concat([x_0, x_1], axis=-1))


def _currin(X: tf.Tensor) -> tf.Tensor:
    x_0 = X[..., 0, tf.newaxis]
    x_1 = X[..., 1, tf.newaxis]
    factor1 = 1 - tf.exp(-1 / (2 * x_1))
    numer = 2300 * x_0 ** 3 + 1900 * x_0 ** (2) + 2092 * x_0 + 60
    denom = 100 * x_0 ** 3 + 500 * x_0 ** (2) + 4 * x_0 + 20
    return factor1 * numer / denom


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


def branincurrin(x: TensorType):
    y1 = _rescaled_branin(x)
    y2 = _currin(x)
    return tf.concat([y1, y2], axis=1)  # minus as we minimize in Trieste


class BraninCurrinGMM(MultiObjectiveTestProblem):
    bounds = [[0, 0], [1, 1]]
    dim = 2

    def objective(self):
        return branincurringmm


def branincurringmm(x):
    y1 = _rescaled_branin(x)
    y2 = _currin(x)
    y3 = gmm_2d(x)
    return tf.concat([y1, y2, y3], axis=-1)


class BraninGMM(MultiObjectiveTestProblem):
    bounds = [[0, 0], [1, 1]]
    dim = 2
    reference_point = tf.constant([200.0, 1.0])
    # reference_point = tf.constant([100.0, 1.0])

    def objective(self):
        return braningmm

    def gen_pareto_optimal_points(self):
        pass


def braningmm(x):
    y1 = _rescaled_branin(x)
    y2 = gmm_2d(x)
    return tf.concat([y1, y2], axis=-1)


class GMMs(MultiObjectiveTestProblem):
    """
    Gaussian Mixture Model introduced
    """

    bounds = [[0, 0], [1, 1]]
    dim = 2

    def objective(self):
        return lambda at: -tf.concat([gmm_2d1(at), gmm_2d2(at)], axis=1)


def gmm_2d1(x, noise_var=0.0):
    """
    from NMES objectives.py
    """
    gmm_pos = tf.constant([[0.2, 0.2], [0.8, 0.2], [0.5, 0.7]], dtype=x.dtype)
    gmm_var = tf.constant([0.20, 0.10, 0.10], dtype=x.dtype) ** 2
    gmm_norm = 2 * pi * gmm_var * tf.constant([0.5, 0.7, 0.7], dtype=x.dtype)
    gaussians = [
        tfp.distributions.MultivariateNormalDiag(
            loc=gmm_pos[i], scale_diag=tf.tile(tf.sqrt(gmm_var[i, tf.newaxis]), [2])
        )
        for i in range(gmm_var.shape[0])
    ]
    f = [gmm_norm[i] * g.prob(x)[:, tf.newaxis] for i, g in enumerate(gaussians)]
    f = tf.reduce_sum(tf.concat(f, axis=1), axis=1, keepdims=True)
    return f


def gmm_2d2(x, noise_var=0.0):
    """
    from NMES objectives.py
    """
    # x = np.atleast_2d(x)
    gmm_pos = tf.constant([[0.4, 0.2], [0.2, 0.6], [0.3, 0.9]], dtype=tf.float64)
    gmm_var = tf.constant([0.20, 0.10, 0.10], dtype=tf.float64) ** 2
    gmm_norm = 2 * pi * gmm_var * tf.constant([0.5, 0.7, 0.7], dtype=tf.float64)
    gaussians = [
        tfp.distributions.MultivariateNormalDiag(
            loc=gmm_pos[i], scale_diag=tf.tile(tf.sqrt(gmm_var[i, tf.newaxis]), [2])
        )
        for i in range(gmm_var.shape[0])
    ]
    f = [gmm_norm[i] * g.prob(x)[:, tf.newaxis] for i, g in enumerate(gaussians)]
    f = tf.reduce_sum(tf.concat(f, axis=1), axis=1, keepdims=True)

    return f


def currin_exp(x, alpha):
    """Computes the currin exponential function."""
    x1 = x[..., :1]
    x2 = x[..., 1:]
    val_1 = 1 - alpha * tf.exp(-1 / (2 * x2))
    val_2 = (2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60) / (
        100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20
    )
    # print((val_1 * val_2).shape)
    return val_1 * val_2


def branin(x, a=1, b=5.1 / (4 * pi ** 2), c=5 / pi, r=6, s=10, t=1 / (8 * pi)):
    """Computes the Branin function."""
    x1 = x[..., :1]
    x2 = x[..., 1:]
    neg_ret = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * tf.cos(x1) + s
    return -neg_ret


def Branin_CurrinExp(x: TensorType, domain_dim=4) -> TensorType:
    """
    Numerical Function at experiment section of original paper
    [0, 1]^4 -> R
    """
    group_dim = 2
    num_groups = int(domain_dim / group_dim)

    def _eval_highd_branin_func(x, func):
        """
        Evaluates higher dimensional branin function.
        referred from https://github.com/kirthevasank/gp-parallel-ts/blob/6401977b4d87b7471c106c35f2bb788a88fd7db9/utils/syn_functions.py#L163
        """
        ret = 0
        for j in range(num_groups):
            ret += func(x[:, j * group_dim : (j + 1) * group_dim])
        return ret

    y1 = _eval_highd_branin_func(x, branin)
    y2 = _eval_highd_branin_func(x, partial(currin_exp, alpha=1))
    return -tf.concat([y1, y2], axis=1)  # minus as we minimize in Trieste


class BraninCurrinExp(MultiObjectiveTestProblem):
    """
    A function only for plotting
    """

    dim = 4
    bounds = [[0] * 4, [1] * 4]

    def objective(self):
        return Branin_CurrinExp


class DTP2(MultiObjectiveTestProblem):
    # reference_point = tf.constant([150.0, 150.0])
    reference_point = tf.constant([10.0, 20.0])

    def __init__(self, dim: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim
        self._bounds = [[0] + [-1] * (dim - 1), [1] * dim]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> list[list[float]]:
        return self._bounds

    def objective(self):
        return deb_test_problem2_objective

    def gen_pareto_optimal_points(self, n: int, seed=None):
        xs = tf.linspace([0], [1], num=n)
        xs = tf.concat([xs, tf.zeros(shape=(n, self.dim - 1))], axis=1)
        return self.objective()(xs)


def deb_test_problem2_objective(x):
    return tf.concat([test_p2_f1(x), test_p2_f2(x)], axis=1)


class Modified_Test_Problem_2(MultiObjectiveTestProblem):
    """
    Modification of Problem 2 to ease the problem for EGO
    """

    def __init__(self, dim: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim
        self._bounds = [[0] + [-1] * (dim - 1), [1] * dim]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> list[list[float]]:
        return self._bounds

    def objective(self):
        return lambda at: tf.concat([test_modified_p2_f1(at), test_modified_p2_f2(at)], axis=1)

    def gen_pareto_optimal_points(self, n: int, seed=None):
        xs = tf.linspace([0], [1], num=n)
        xs = tf.concat([xs, tf.zeros(shape=(n, self.dim - 1))], axis=1)
        return self.objective()(xs)


class Test_Problem_3(MultiObjectiveTestProblem):
    """
    Original Test problem from Deb's paper
    """

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim
        assert dim >= 2
        self._bounds = [[0, 0] + [-1] * (dim - 2), [1, 1] + [1] * (dim - 2)]

    def objective(self):
        return lambda at: tf.concat([test_p3_f1(at), test_p3_f2(at)], axis=1)


class MDTP3(MultiObjectiveTestProblem):
    reference_point = tf.constant([10.0, 10.0])

    def __init__(self, dim: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim
        assert dim >= 2
        self._bounds = [[0, 0] + [-1] * (dim - 2), [1, 1] + [1] * (dim - 2)]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> list[list[float]]:
        return self._bounds

    def objective(self):
        # return lambda at: tf.concat([test_modified_p3_f1(at), test_modified_p3_f2(at)], axis=1)
        return modified_deb_test_problem3_objective
        # Note: 0.723 is just a munal shift of its mean to rougly 0

    def gen_pareto_optimal_points(self):
        pass


def modified_deb_test_problem3_objective(x):
    return tf.concat([test_modified_p3_f1(x), test_modified_p3_f2(x)], axis=1)


class Test_Problem_4(MultiObjectiveTestProblem):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim
        assert dim >= 2
        self._bounds = [[0, -0.15] + [-1] * (dim - 2), [1, 1] + [1] * (dim - 2)]

    def objective(self):
        return lambda at: tf.concat([test_p4_f1(at), test_p4_f2(at)], axis=1)


class Test_Problem_5(MultiObjectiveTestProblem):
    """
    A self designed multi-objective problem
    obj1 is Sin + Linear function from NMES paper
    obj2 is a modified Sin+ Linear function
    """

    def __init__(self, dim=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim
        self._bounds = [[0] * 2, [1] * 2]

    def objective(self):
        return lambda at: tf.concat([test_p5_f1(at), test_p5_f2(at)], axis=1)


def test_p2_f1(x):
    return x[..., 0, tf.newaxis]


def test_p2_f2(x):
    def h(x1):
        return 1 - x1 ** 2

    def S(x1, alpha=1, beta=10):
        return alpha / (0.2 + x1) + beta * x1 ** 2

    def g(x):
        """
        Only need when dim of x > 2
        """
        xi = x[:, 1:]
        return tf.reduce_sum(10 + xi ** 2 - 10 * tf.cos(4 * pi * xi), axis=1, keepdims=True)

    x_1 = x[..., 0, tf.newaxis]
    return h(x_1) + g(x) * S(x_1)


def test_modified_p2_f1(x):
    return x[..., 0, tf.newaxis]


def test_modified_p2_f2(x):
    def h(x1):
        return 1 - x1 ** 2

    def S(x1, alpha=1, beta=10):
        return alpha / (0.2 + x1) + beta * x1 ** 2

    def g(x):
        """
        Only need when dim of x > 2
        """
        xi = x[:, 1:]
        return tf.reduce_sum(10 + xi ** 2 - 10 * tf.cos(4 * pi * xi), axis=1, keepdims=True)

    x_1 = x[..., 0, tf.newaxis]
    return h(x_1) + g(x) * S(x_1)


def test_p3_f1(x):
    x1 = x[..., 0, tf.newaxis]
    return x1


def test_p3_f2(x):
    def h(x):
        x2 = x[..., 1, tf.newaxis]
        return 2 - 0.8 * tf.exp(-(((x2 - 0.35) / 0.25) ** 2)) - tf.exp(-(((x2 - 0.85) / 0.03) ** 2))
        # tf.exp(- ((x2 - 0.85) / 0.04) ** 2)

    def S(x):
        return 1 - tf.sqrt(x[..., 0, tf.newaxis])

    def g(x):
        """
        Only need when dim of x > 2
        """
        x3 = x[..., 2:]
        return 50 * tf.reduce_sum(x3 ** 2, keepdims=True, axis=1)

    return h(x) * (S(x) + g(x))


def test_modified_p3_f1(x):
    return x[..., 0, tf.newaxis]


def test_modified_p3_f2(x):
    def h(x):
        x2 = x[..., 1, tf.newaxis]
        return (
            1 - 0.9 * tf.exp(-(((x2 - 0.8) / 0.1) ** 2)) - 1.3 * tf.exp(-(((x2 - 0.3) / 0.03) ** 2))
        )
        # return 2 - 1 * tf.exp(- ((x2 - 0.8) / 0.1) ** 2) - \
        #        1.3 * tf.exp(- ((x2 - 0.3) / 0.02) ** 2)

    def S(x):
        # return 1.2 - tf.sqrt(x[..., 0, tf.newaxis])
        # return 2 - tf.sqrt(x[..., 0, tf.newaxis])
        return -x[..., 0, tf.newaxis]

    def g(x):
        """
        Only need when dim of x > 2
        """
        x3 = x[..., 2:]
        return 50 * tf.reduce_sum(x3 ** 2, keepdims=True, axis=1)

    return h(x) + (S(x) + g(x))


def test_p4_f1(x):
    """
    y=X
    """
    return x[..., 0, tf.newaxis]


def test_p4_f2(x):
    def h(x):
        x1 = x[..., 0, tf.newaxis]
        x2 = x[..., 1, tf.newaxis]
        return (
            2
            - x1
            - 0.8 * tf.exp(-(((x1 + x2 - 0.35) / 0.25) ** 2))
            - tf.exp(-(((x1 + x2 - 0.85) / 0.03) ** 2))
        )

    def S(x):
        return 1 - tf.sqrt(x[..., 0, tf.newaxis])

    def g(x):
        """
        Only need when dim of x > 2
        """
        x3 = x[..., 2:]
        return 50 * tf.reduce_sum(x3 ** 2, keepdims=True, axis=1)

    return h(x) * (S(x) + g(x))


def test_p5_f1(x):
    x1 = x[..., 0, tf.newaxis]
    return x1 ** 2


def test_p5_f2(x):
    x2 = x[..., 1, tf.newaxis]
    x1 = x[..., 0, tf.newaxis]
    return -(tf.sin(5 * pi * x2 ** 2) + 0.5 * x2) - x1


class SinLinForrester(MultiObjectiveTestProblem):
    dim = 1
    bounds = [[0], [1]]
    # reference_point = tf.constant([2.0, 15.0])
    reference_point = tf.constant([1.0, 16.0])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def objective(self):
        return sin_lin_forrester_objective

    def gen_pareto_optimal_points(self):
        pass


def sin_lin_forrester_objective(x):
    """
    This is a helper function used in the sense that multiprocessing cant pickle lambda function
    """
    return tf.concat([synthetic_1d_01(x), forrester(x)], axis=1)


def synthetic_1d_01(x):
    """
    from NMES paper
    """
    f = tf.sin(5 * pi * x ** 2) + 0.5 * x
    return -f


def forrester(x):
    return (6 * x - 2) ** 2 * tf.sin(12 * x - 4)


if __name__ == "__main__":
    import tensorflow as tf

    a = branincurringmm(tf.constant([[0.20, 0.30], [0.5, 0.1]]))
    # WFG problem
    # func = BraninCurrin().objective()
#
#
# def branin__(at):
#     return tf.gather(func(at), [1], axis=1)
#
#
# from PyOptimize.utils.visualization import view_2D_function_in_contour
#
# view_2D_function_in_contour(branin__, [[0, 1]] * 2, show=True)
