import math
from typing import Callable, List, Optional, Union

import tensorflow as tf

from ..acquisition.multi_objective.pareto import Pareto
from ..types import TensorType


def log_HV_differenceV2(
    recommendation_input: TensorType,
    true_func: Callable,
    ref_point: Union[TensorType, List[float]],
    true_pf: Optional[TensorType] = None,
    ideal_hv: Optional[TensorType] = None,
):

    observations = true_func(recommendation_input)
    ref_point = tf.convert_to_tensor(ref_point, dtype=observations.dtype)
    if ideal_hv is None:
        ideal_hv = Pareto(true_pf).hypervolume_indicator(ref_point)
    screened_obs = observations[tf.reduce_all(observations <= ref_point, -1)]
    if tf.size(screened_obs) == 0:
        obs_hv = 0.0
    else:
        obs_hv = Pareto(screened_obs).hypervolume_indicator(ref_point)
    tf.print(ideal_hv - obs_hv)
    return math.log10(ideal_hv - obs_hv)


def log_HV_difference(
    observations: TensorType,
    ref_point: Union[TensorType, List[float]],
    true_pf: Optional[TensorType] = None,
    ideal_hv: Optional[TensorType] = None,
):
    """
    Log HV difference, refer botorch
    :param observations
    :param ref_point
    :param true_pf
    :param ideal_hv if ideal_hv is provided, use it as the groundtruth
    """
    ref_point = tf.convert_to_tensor(ref_point, dtype=observations.dtype)
    if ideal_hv is None:
        ideal_hv = Pareto(true_pf).hypervolume_indicator(ref_point)
    obs_hv = Pareto(observations).hypervolume_indicator(ref_point)
    tf.print(ideal_hv - obs_hv)
    return math.log10(ideal_hv - obs_hv)


def HV(observations: TensorType, ref_point: TensorType):
    """
    Hypervolume Indifcator
    :param observations
    :param ref_point
    """
    return Pareto(observations).hypervolume_indicator(ref_point)


def GD_plus(observations: TensorType, true_pf: TensorType):
    """
    GD+ metric,
    refer https://pymoo.org/misc/performance_indicator.html
    """
    A = observations.shape[0]
    gd_plus = 0.0
    for popts in observations:
        nearest_idx = tf.argmin(tf.norm(popts - true_pf, ord="euclidean", axis=1))
        pairwise_max = tf.maximum(popts, true_pf[nearest_idx])  # element wise max
        d_plus = tf.sqrt(tf.reduce_sum((pairwise_max - true_pf[nearest_idx]) ** 2)) / A
        gd_plus += d_plus
    return gd_plus


def IGD_plus(observations: TensorType, true_pf: TensorType):
    """
    IGD+ metric, which is weak Pareto complaint
    refer: https://pymoo.org/misc/performance_indicator.html
    """
    Z = true_pf.shape[0]
    igd_plus = 0.0
    for popts in true_pf:
        # find the nearest point in observations
        nearest_idx = tf.argmin(tf.norm(observations - popts, ord="euclidean", axis=1))
        pairwise_max = tf.maximum(popts, observations[nearest_idx])
        d_plus = tf.sqrt(tf.reduce_sum((pairwise_max - popts) ** 2)) / Z
        igd_plus += d_plus
    return igd_plus


def GDp(observations: TensorType, true_pf: TensorType, p=2):
    """
    :cite: schutze2012using Eq. 11
    """
    N = observations.shape[0]
    gd_plus = 0.0
    for popts in observations:
        nearest_idx = tf.argmin(tf.norm(popts - true_pf, ord="euclidean", axis=1))
        # pairwise_max = tf.maximum(popts, true_pf[nearest_idx])  # element wise max
        d_plus = (tf.norm(popts - true_pf[nearest_idx], ord="euclidean") ** p) / N
        gd_plus += d_plus
    return gd_plus ** (1 / p)


def IGDp(observations: TensorType, true_pf: TensorType, p=2):
    """
    :cite: schutze2012using Eq. 35
    """
    N = true_pf.shape[0]
    igd_plus = 0.0
    for popts in true_pf:
        # find the nearest point in observations
        nearest_idx = tf.argmin(tf.norm(observations - popts, ord="euclidean", axis=1))
        # pairwise_max = tf.maximum(popts, observations[nearest_idx])
        d_plus = (tf.norm(observations[nearest_idx] - popts, ord="euclidean") ** p) / N
        igd_plus += d_plus
    return igd_plus ** (1 / p)


def AverageHausdoff(
    recommendation_input: TensorType,
    true_func: Callable,
    true_pf: TensorType,
    p=2,
    scaler: [tuple, None] = None,
):
    """
    :cite: schutze2012using Eq. 45

    The scaler is used for scaling each obj to roughly the same range [0, 1]^K, so that
    euclidean based metric won't bias on certain obj(e.g., if y1 ranges [0, 1] and y2 ranges [0, 1000],
    then if not scale there might be a biase)
    """
    if tf.equal(tf.size(recommendation_input), 0):
        return 1e10

    obj_obs = true_func(recommendation_input)

    if hasattr(true_func, "constraint"):
        raise NotImplementedError
    if tf.equal(tf.size(obj_obs), 0):  # empty obs after penalize
        return 1e10
    if scaler is not None:
        scaler_lb = tf.convert_to_tensor(scaler[0], dtype=true_pf.dtype)
        scaler_ub = tf.convert_to_tensor(scaler[1], dtype=true_pf.dtype)
        true_pf = (true_pf - scaler_lb) / (scaler_ub - scaler_lb)
        obj_obs = (obj_obs - scaler_lb) / (scaler_ub - scaler_lb)
    return tf.maximum(GDp(obj_obs, true_pf, p), IGDp(obj_obs, true_pf, p))


def Infer_regret_penalized_on_var_constraint(
    recommendation_input: TensorType,
    true_func: Callable,
    true_optimum: TensorType,
    variance_threshold: float,
    penalization_factor: int = 10,
):
    mean, var = tf.split(true_func(recommendation_input), 2, -1)
    return tf.squeeze(
        mean - true_optimum + penalization_factor * tf.maximum(var - variance_threshold, 0.0)
    )


def Utility_gap(
    recommendation_input: TensorType,
    true_func: Callable,
    true_optimum: TensorType,
    worst_function_val_in_design_space: TensorType,
    variance_threshold: float,
):
    mean, var = tf.split(true_func(recommendation_input), 2, -1)
    if var > variance_threshold:
        return worst_function_val_in_design_space - true_optimum
    else:
        return tf.squeeze(mean - true_optimum)


def Infer_regret(recommendation_input: TensorType, true_func: Callable, true_optimum: TensorType):
    return tf.squeeze(true_func(recommendation_input) - true_optimum)


def Scalarization_Infer_regret(
    recommendation_input: TensorType,
    true_func: Callable,
    true_optimum: TensorType,
    scalarize_alpha: float,
):
    mean, var = tf.split(true_func(recommendation_input), 2, -1)
    return tf.squeeze(mean + scalarize_alpha * var - true_optimum)


def regret(observations: TensorType, optimum: TensorType):
    """
    regret for minimization
    """
    return tf.squeeze(observations - optimum)


def current_best(value):
    return tf.squeeze(value)
