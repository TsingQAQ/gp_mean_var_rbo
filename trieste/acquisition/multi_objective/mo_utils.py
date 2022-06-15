from __future__ import annotations

import tensorflow as tf
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from tensorflow_probability import distributions as tfd

from trieste.models import ModelStack
from trieste.space import Box
from trieste.types import TensorType


def moo_optimize_pymoo(
    f,
    input_dim: int,
    obj_num: int,
    bounds: tuple,
    popsize: int,
    num_generation: int,
    return_pf_x: bool = False,
    cons=None,
    cons_num: float = 0,
) -> [TensorType, None, [None, None]]:
    """
    Multi-Objective Optimizer by pymoo

    :param f
    :param input_dim
    :param bounds
    :param popsize
    :param num_generation
    :param return_pf_x
    :param cons
    :param cons_num

    :return if no feasible pareto frontier has been located, return None or [None, None]
    """

    if cons is not None:
        assert cons_num > 0

    def func(x):
        return f(tf.convert_to_tensor(x)).numpy()

    def cfunc(x):
        return cons(tf.convert_to_tensor(x)).numpy()

    class MyProblem(Problem):
        def __init__(self, n_var, n_obj, n_constr: int = cons_num):
            super().__init__(
                n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=bounds[0], xu=bounds[1]
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = func(x)
            if cons_num > 0:
                out["G"] = -cfunc(x)  # in pymoo, by default <0 is feasible

    problem = MyProblem(n_var=input_dim, n_obj=obj_num)
    algorithm = NSGA2(
        pop_size=popsize,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, ("n_gen", num_generation), save_history=False, verbose=False)
    if return_pf_x:
        return res.F, res.X
    return res.F


def moo_optimize_pygmo(
    f,
    input_dim: int,
    obj_num: int,
    bounds: tuple,
    popsize: int,
    num_generation: int = 1000,
    return_pf_x: bool = False,
    cons=None,
    cons_num: float = 0,
):
    """
    Multi-Objective Optimizer by pygmo library, used in MESMO
    """
    from platypus import NSGAII, Problem, Real

    if cons is not None:
        assert cons_num > 0

    def helper_obj(x):
        """"""
        if cons is not None:
            return (
                tf.squeeze(f(tf.convert_to_tensor(x)[tf.newaxis, ...])).numpy(),
                tf.squeeze(cons(tf.convert_to_tensor(x)[tf.newaxis, ...])).numpy(),
            )
        return tf.squeeze(f(tf.convert_to_tensor(x)[tf.newaxis, ...])).numpy()

    problem = Problem(nvars=input_dim, nobjs=obj_num, nconstrs=cons_num)
    for d_idx in range(input_dim):
        problem.types[d_idx] = Real(bounds[0][d_idx], bounds[1][d_idx])
    problem.function = helper_obj
    problem.constraints[
        :
    ] = ">=0"  # refer https://platypus.readthedocs.io/en/latest/getting-started.html
    algorithm = NSGAII(problem, population_size=popsize)
    algorithm.run(num_generation)

    if cons is None and cons_num == 0:
        algorithm_result = algorithm.result
    else:
        algorithm_result = []
        for solution in algorithm.result:
            if tf.reduce_all(tf.convert_to_tensor(solution.constraints) >= 0):
                algorithm_result.append(solution)
    if not algorithm_result:
        return None if return_pf_x is False else (None, None)
    pareto_front = tf.convert_to_tensor(
        [list(solution.objectives) for solution in algorithm_result]
    )
    if return_pf_x:
        pareto_front_x = tf.convert_to_tensor(
            [list(solution.variables) for solution in algorithm_result]
        )
        return pareto_front, pareto_front_x
    else:
        return pareto_front


def inference_pareto_fronts_from_gp_mean(
    models: ModelStack,
    search_space: Box,
    popsize: int = 20,
    num_moo_iter: int = 500,
    cons_models: [ModelStack, None] = None,
    min_feasibility_probability=0.5,
    discrete_input: bool = False,
) -> (TensorType, TensorType):
    """
    get the pareto frontier inferenced by GP posterior mean
    """

    def mean_obj(at):
        return models.predict(at)[0]

    # note we wanna the aggregated PoF > min_feasibility_probability
    def mean_con(at):
        # cons_models.predict(at)[0]
        mean, var = cons_models.predict(at)
        prob_fea = tf.reduce_prod(1 - tfd.Normal(mean, tf.sqrt(var)).cdf(0.0), -1, keepdims=True)
        # prob_fea = 1 - tfd.Normal(mean, tf.sqrt(var)).cdf(0.0)
        # print(prob_fea)
        penalty_factor = 1.0
        return (prob_fea - min_feasibility_probability) * penalty_factor

    if not discrete_input:
        pf, pfx = moo_optimize_pymoo(
            mean_obj,
            input_dim=len(search_space.lower),
            obj_num=len(models._models),
            bounds=(search_space.lower.numpy(), search_space.upper.numpy()),
            popsize=popsize,
            num_generation=num_moo_iter,
            return_pf_x=True,
            cons=mean_con if cons_models is not None else None,
            # cons_num=len(cons_models._models) if cons_models is not None else 0,
            cons_num=1 if cons_models is not None else 0,
        )
        return pf, pfx

    else:
        raise NotImplementedError
        # use a uniformed grid to approximate the expensive PF optimize
        # however, this seems only support input_dim <= 3 as otherwise may lead OOM
        lb, ub = search_space.lower, search_space.upper
        xs = tf.linspace(lb, ub, 1000)
        # TODO: get all combinations of xs through dim 2
