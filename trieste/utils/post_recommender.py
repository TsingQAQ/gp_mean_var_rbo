"""
Different post processing method
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Mapping

import tensorflow as tf

from ..acquisition.multi_objective.mo_utils import inference_pareto_fronts_from_gp_mean
from ..acquisition.multi_objective.pareto import non_dominated
from ..acquisition.optimizer import automatic_optimizer_selector
from ..bayesian_optimizer import OptimizationResult
from ..data import Dataset
from ..models.interfaces import ModelStack
from ..observer import CONSTRAINT, OBJECTIVE
from ..space import Box, SearchSpace
from ..types import TensorType


def probability_smaller(x, x_prime, probability_evaluator) -> TensorType:
    """
    calculate the probability that Pr(f(x)<f(x_prime))
    :param probability_evaluator: return mean and variance
    :param x
    :param, x_prime
    """
    tf.debugging.assert_shapes(
        [(x_prime, [1, "num"])],
        message="points must be of shape [1,D]",
    )
    mean_x, var_x = probability_evaluator(x)
    mean_x_prime, var_x_prime = probability_evaluator(x_prime)
    return 0.5 * (1 + tf.math.erf((mean_x_prime - mean_x) / (var_x + var_x_prime)))


def pareto_front_on_mean_based_on_data(
    models: Mapping[str, ModelStack],
    datas: Mapping[str, Dataset] | TensorType,
    min_feasibility_probability: float = 0.5,
    constraint_tag: [str, None] = None,
) -> TensorType:
    """
    get model posterior mean predicted pareto front from the provided data

    assume >0 is feasible
    :param models
    :param datas
    :param min_feasibility_probability
    :param constraint_tag
    """
    if constraint_tag is None:
        _, dominance = non_dominated(models[OBJECTIVE].predict(datas)[0])
        return tf.gather_nd(datas, tf.where(tf.equal(dominance, 0)))

    # In case there are constraints
    else:
        obs = models[OBJECTIVE].predict(datas[OBJECTIVE].query_points)[0]
        cons = models[CONSTRAINT].predict(datas[CONSTRAINT].observations)[0]
        feasible_index = tf.where(tf.reduce_all(cons > 0.0, -1))
        # calculate the dominance through feasible query points
        feasible_obs = tf.gather_nd(obs, feasible_index)
        nd_feasible_obs, _ = non_dominated(feasible_obs)
        # get index of non-dominated feasible obs
        # [N, D] & [M, D] -> [N, M, D] -> [M, D]
        fea_nd_idx = tf.reduce_any(
            tf.reduce_all(tf.equal(obs, tf.expand_dims(nd_feasible_obs, -2)), -1), 0
        )
        return tf.gather_nd(datas[OBJECTIVE].query_points, tf.where(fea_nd_idx))


def feasible_mean_under_variance_constraint(
    model, datas: Mapping[str, Dataset] | TensorType, variance_threshold: float
) -> TensorType:
    means, vars = tf.split(model(datas), 2, -1)
    feasible_mask = vars <= variance_threshold

    penalized_means = tf.zeros_like(means)
    penalized_means += tf.cast(feasible_mask, dtype=datas.dtype) * means
    penalized_means += tf.cast(~feasible_mask, dtype=datas.dtype) * 1000.0
    # return tf.expand_dims(datas[tf.argmin(penalized_means)[0]], -1) # [1, 1]
    return tf.gather(datas, tf.argmin(penalized_means))


def mean_var_pareto_front_on_mean_based_on_data(
    mean_var_func: Callable,
    datas: Mapping[str, Dataset] | TensorType,
    min_feasibility_probability: float = 0.5,
    constraint_tag: [str, None] = None,
) -> TensorType:
    """
    get model posterior mean predicted pareto front from the provided data

    assume >0 is feasible
    :param mean_var_func
    :param datas
    :param min_feasibility_probability
    :param constraint_tag
    """
    assert constraint_tag is None
    _, dominance = non_dominated(mean_var_func(datas))
    return tf.gather_nd(datas, tf.where(tf.equal(dominance, 0)))


def mean_var_scalarization_on_mean_based_on_data(
    mean_var_func: Callable, datas: Mapping[str, Dataset] | TensorType, scalarization_alpha: float
):
    mean, var = tf.split(mean_var_func(datas), 2, -1)
    recommend_x = tf.gather_nd(datas, tf.argmin(mean + scalarization_alpha * var))
    return recommend_x if tf.rank(recommend_x) > 1 else recommend_x[tf.newaxis]


def min_on_mean_based_on_data(model, data: TensorType) -> TensorType:
    """"""
    return tf.gather_nd(data, tf.argmin(model.predict(data)[0]))[tf.newaxis, ...]


def min_on_mean_based_on_search_space(
    model, data: TensorType, search_space: SearchSpace
) -> TensorType:
    """
    Get min recommendation based on minimizing model predict mean
    """
    starting_x0 = tf.gather_nd(data, tf.argmin(model.predict(data)[0]))[tf.newaxis, ...]

    def neg_pred(x):
        """
        The x some times will be rank 3, where dim 1 is the batch dim, in this cases
        :param: x [Batch_shape, 1, dim]
        """
        if tf.rank(x) == 3:
            x = tf.squeeze(x, 1)
        return -model.predict(x)[0]

    opt_x = automatic_optimizer_selector(search_space, neg_pred)

    return starting_x0 if model.predict(starting_x0)[0] < model.predict(opt_x)[0] else opt_x


def min_on_mean_with_var_constraint_based_on_data(
    optimization_result: OptimizationResult, min_feasibility: float = 0.5
) -> tuple[TensorType, TensorType]:
    models = optimization_result.try_get_final_models()
    datasets = optimization_result.try_get_final_datasets()
    oracle_data = optimization_result.try_get_pending_oracle()
    candidate = (
        tf.convert_to_tensor(oracle_data)
        if oracle_data is not None
        else datasets[OBJECTIVE].query_points
    )

    acq_builder = optimization_result.final_acquisition_builder

    objective_model = models[OBJECTIVE]

    constraint_fn = acq_builder._constraint_builder.prepare_acquisition_function(datasets, models)
    pof = constraint_fn(candidate[:, None, ...])
    is_feasible = tf.squeeze(pof >= min_feasibility, axis=-1)

    if not tf.reduce_any(is_feasible):
        raise ValueError(f"No design satisfying the constraints. PoF: {pof}")

    feasible_query_points = tf.boolean_mask(candidate, is_feasible)
    feasible_mean, _ = objective_model.predict(feasible_query_points)
    preferable_sort_idx = tf.argsort(feasible_mean, 0)
    return (
        feasible_query_points[tf.squeeze(preferable_sort_idx[0])],
        feasible_mean[tf.squeeze(preferable_sort_idx[0])],
    )


def min_on_mean_plus_var_based_on_data(
    model, data: TensorType, search_space: SearchSpace, coefficient: float = 1
):
    """
    Get the min recommendation based on minimizing the model predict:
    mean + coefficient * variance
    """
    # TODO:
    pass


def min_on_mean_plus_var_based_on_search_space(
    model, data: TensorType, search_space: SearchSpace, coefficient: float = 1
):
    """
    Get the min recommendation based on minimizing the model predict:
    mean + coefficient * variance
    """
    # TODO:
    pass


def prob_min_based_on_search_space(
    model, data: TensorType, search_space: SearchSpace
) -> TensorType:
    """"""
    starting_x0 = tf.gather_nd(data, tf.argmin(model.predict(data)[0]))[tf.newaxis, ...]
    cost_func = lambda x: partial(
        probability_smaller, x_prime=starting_x0, probability_evaluator=model.predict
    )(tf.squeeze(x, 1))
    opt_x = automatic_optimizer_selector(search_space, cost_func)
    return starting_x0 if probability_smaller(opt_x, starting_x0, model.predict) > 0.5 else opt_x


def pareto_front_on_mean_based_on_model(
    search_space: Box,
    models: Mapping[str, ModelStack],
    data: TensorType,
    kwargs_for_inferred_pareto={},
    min_feasibility_probability: float = 0.5,
    discrete_input: bool = False,
) -> tuple[TensorType, TensorType]:
    """
    extract Pareto front from posterior mean of GP model.

    The following is from PESMOC paper:
    If constraint has present: we consider that a constraint is satisfied at an input
    location x if the probability that the constraint is larger than zero is above 1 − δ
    where δ is 0.05. That is, p(c j (x ≥ 0) ≥ 1 − δ. When no feasible solution is found,
    we simply return the points that are most likely to be feasible by itera-
    tively increasing δ in 0.05 units.

    :param discrete_input:
    """
    while min_feasibility_probability >= 0:
        res, resx = inference_pareto_fronts_from_gp_mean(
            models[OBJECTIVE],
            search_space,
            popsize=kwargs_for_inferred_pareto.get("popsize", 20),
            cons_models=models[CONSTRAINT] if CONSTRAINT in models.keys() else None,
            num_moo_iter=kwargs_for_inferred_pareto.get("num_moo_iter", 500),
            min_feasibility_probability=min_feasibility_probability,
            discrete_input=discrete_input,
        )
        # print(res.shape)
        if resx is None:
            min_feasibility_probability -= 0.05
            print(
                f"no satisfied constraint PF found, "
                f"decrease min_feasible_prob to: {min_feasibility_probability}"
            )
        else:
            return resx
