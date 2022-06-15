from typing import Callable

from trieste.acquisition.function.function import SingleModelAcquisitionBuilder
from trieste.models.config import ModelConfig


def ff_iu_serial_benchmarker(
    file_identifier: int,
    *,
    acq,
    constraint_acq: [SingleModelAcquisitionBuilder, None] = None,
    num_obj: int = 1,
    kwargs_for_acquisition_rule: dict = {},
    kwargs_for_optimizer: dict = {},
    kwargs_for_optimize: dict = {},
    kwargs_for_acq: dict = {},
    kwargs_for_kernel: dict = {},
    kwargs_for_constraint_acq: dict = {},
    ref_point_setting: str = "default",
    kernel_jitter: [None, float] = None,
    post_profiler: Callable = None,
    q: int = 1,
    is_return: bool = True,
    save_result_to_file: bool = True,
    path_prefix: str = "",
    custom_observer=None,
    **kwargs,
):
    """
    Sequential benchmarker of robust Bayesian Optimization framework
    :param file_identifier
    :param acq acquisition function
    :param constraint_acq
    :param num_obj: objective number
    :kwargs is_normal_input_uncertainty: whether the input uncertainty (if have) is normal distributed
    :kwargs noise_type
    :kwargs methods:
    :kwargs is_normal_input_uncertainty:
    :param kwargs_for_optimize
    :param kwargs_for_kernel
    :param kwargs_for_acq
    :param kernel_jitter: a white kernel with jitter as variance, used for stability to avoid Cholesky issue
    :param ref_point_setting: "default" or "problem specified"
    :param post_profiler: callable function that has input: (mo_result, problem_individual_name),
        which is design for usage when one would like to save results to file (more used in synthetic results)
    :param is_return
    :param parallel
    :param q Batch size
    :param save_result_to_file
    :param path_prefix
    :param kwargs_for_constraint_acq
    :param custom_observer
    :param recommender: final recommender of shown what to be used in applications
    """

    import os

    import gpflow
    import numpy as np
    import tensorflow as tf
    from gpflow import default_float
    from tensorflow_probability import distributions as tfd

    import trieste
    from trieste.acquisition.rule import ActiveLearningAcquisitionRule, Random
    from trieste.bayesian_optimizer import (
        BayesianOptimizer,
        EfficientGlobalOptimization,
        RBOTwoStepBayesianOptimizer_MeanVar,
    )
    from trieste.data import Dataset
    from trieste.models import create_model
    from trieste.models.interfaces import ModelStack
    from trieste.objectives import multi_objectives, single_objectives
    from trieste.observer import OBJECTIVE

    # ---------------------------------- Settings -----------------------------------
    assert num_obj == 1
    num_objective = num_obj
    file_identifier_str = str(file_identifier)
    # If there is seed in kwargs, change this seed to be a unique but reproducible seed for each child process
    if "kwargs_for_objective" in kwargs and "seed" in kwargs["kwargs_for_objective"]:
        kwargs["kwargs_for_objective"]["seed"] = (
            kwargs["kwargs_for_objective"]["seed"] + file_identifier * 10
        )
    # Unpack settings
    if "benchmark" not in kwargs:
        if num_objective == 1:
            func_inst = getattr(single_objectives, kwargs["benchmark_name"])(
                **kwargs.get("kwargs_for_objective", {})
            )
        else:
            func_inst = getattr(multi_objectives, kwargs["benchmark_name"])(
                **kwargs.get("kwargs_for_objective", {})
            )
    else:  # this is the hard code only for matlab benchmark
        func_inst = kwargs["benchmark"]

    func = func_inst.objective()
    bounds = func_inst.bounds
    dim_benchmark = func_inst.dim

    total_iter = kwargs.get("total_iter", 20)  # This is used to control the maximum func eval
    tf.debugging.assert_greater_equal(total_iter, 0)

    # -----------Define Obj & search space ----
    if custom_observer is not None:
        # Useful when you have constraint
        observer = custom_observer
    else:  # Only Objective Optimization
        observer = trieste.objectives.utils.mk_observer(func, OBJECTIVE)

    search_space = trieste.space.Box(bounds[0], bounds[1])
    if kwargs.get("initial_x") is not None:
        print("init x provided, evaluate on it as doe")
        if kwargs["initial_x"] == "Stored":
            # note: hard code
            print(f"use stored initial x: xs_{file_identifier_str}.txt")
            try:
                xs = np.loadtxt(
                    os.path.join(
                        "..",
                        "cfg",
                        "initial_xs",
                        kwargs["benchmark_name"],
                        f"xs_{file_identifier_str}.txt",
                    )
                )
            except:
                xs = np.loadtxt(
                    os.path.join(
                        ".",
                        "cfg",
                        "initial_xs",
                        kwargs["benchmark_name"],
                        f"xs_{file_identifier_str}.txt",
                    )
                )
            if dim_benchmark == 1:
                xs = np.atleast_2d(xs).T
        else:
            xs = kwargs["initial_x"]
        xs = tf.convert_to_tensor(xs, dtype=tf.float64)
        datasets = observer(xs)
    else:
        assert "doe_num" in kwargs, ValueError("doe_num must be specified if no init x provided")
        doe_num = kwargs["doe_num"]
        num_initial_points = doe_num
        if kwargs.get("acq_name") in ["MO-MVA-BO", "CO-MVA-BO"]:  # descrite sample
            x_init = search_space.discretize_v2(kwargs_for_acq["tau"]).sample(num_initial_points)
        else:
            x_init = search_space.sample(num_initial_points)
        datasets = observer(x_init)

    # --------------------------------------------
    def create_vanilla_gp_model(data, input_dim, init_lengthscale=1.0):
        """
        Create Standard GP Model
        """
        variance = tf.math.reduce_variance(data.observations)
        lengthscale = init_lengthscale * np.ones(input_dim, dtype=default_float())
        kernel = gpflow.kernels.SquaredExponential(variance=variance, lengthscales=lengthscale)
        if "lengthscales_prior" in kwargs_for_kernel:
            kernel.lengthscales.prior = kwargs_for_kernel["lengthscales_prior"]
        if "kernel_variance_prior" in kwargs_for_kernel:
            # FIXME: This has some issue, abort atm
            if kwargs_for_kernel["kernel_variance_prior"] == "Weak_LogNormal":
                kernel.variance.prior = tfd.LogNormal(
                    tf.math.log(tf.constant([1], dtype=tf.float64)),
                    tf.constant([0.5], dtype=tf.float64),
                )
            else:
                kernel.variance.prior = kwargs_for_kernel["kernel_variance_prior"]
        if kernel_jitter is not None:
            jitter = gpflow.kernels.White(kernel_jitter)
            kernel = kernel + jitter
        gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1.1e-6)
        gpflow.set_trainable(gpr.likelihood, False)
        return create_model(
            ModelConfig(
                **{
                    "model": gpr,
                    "optimizer": gpflow.optimizers.Scipy(),
                    "optimizer_args": {
                        "minimize_args": {"options": dict(maxiter=200)},
                    },
                }
            )
        )

    # Standard BO
    objective_models = create_vanilla_gp_model(
        Dataset(
            datasets[OBJECTIVE].query_points,
            datasets[OBJECTIVE].observations,
        ),
        input_dim=dim_benchmark,
    )

    models = {OBJECTIVE: objective_models}

    if ref_point_setting == "default":
        _acq = acq(**kwargs_for_acq).using(OBJECTIVE)
    elif ref_point_setting == "problem specified":
        if not constraint_acq:
            # we assert acq has a key word "reference_point_setting" to set the reference point
            _acq = acq(**kwargs_for_acq, reference_point_setting=func_inst.reference_point).using(
                OBJECTIVE
            )
        else:
            raise NotImplementedError
            # con_acq_builder = constraint_acq(**kwargs_for_constraint_acq).using(VARIANCE_CONSTRAINT)
            # _acq = acq(
            #     **kwargs_for_acq,
            #     reference_point_setting=func_inst.reference_point,
            #     constraint_builder=con_acq_builder,
            # )
    else:
        raise ValueError(f"ref point setting {ref_point_setting} not understood")

    if kwargs.get("acq_name") == "Random":
        _rule = Random(builder=_acq, num_query_points=q)
    elif kwargs.get("acq_name") in ["MO-MVA-BO", "CO-MVA-BO"]:
        _rule = ActiveLearningAcquisitionRule(builder=_acq)
    else:
        _rule = EfficientGlobalOptimization(
            builder=_acq, num_query_points=q, **kwargs_for_acquisition_rule
        )

    if kwargs.get("acq_name") in ["MO-MVA-BO", "CO-MVA-BO", "MT-MVA-BO", "US"]:
        if kwargs_for_optimizer["kw_al"]["which_al_obj"] == "P_std":  # Hard Code
            kwargs_for_optimizer["kw_al"]["base_dist"] = kwargs["seed_sampler"]
        bo = RBOTwoStepBayesianOptimizer_MeanVar(observer, search_space, **kwargs_for_optimizer)
    else:
        print(
            f"use default BayesianOptimizer, doesn't accept kwagrs, abort kwargs : {kwargs_for_optimizer}"
        )
        bo = BayesianOptimizer(observer, search_space)

    try:
        mo_result = bo.optimize(
            total_iter, datasets, models, acquisition_rule=_rule, **kwargs_for_optimize
        )
        recommenders_history = post_profiler(mo_result)
        if save_result_to_file:
            # Save metric score
            for recommender_name in recommenders_history.keys():
                for metric_name, metric_val in recommenders_history[recommender_name].items():
                    _path = os.path.join(
                        path_prefix, kwargs.get("acq_name"), kwargs["file_info_prefix"]
                    )
                    _file_name = "_".join(
                        [
                            kwargs["benchmark_name"],
                            file_identifier_str,
                            recommender_name,
                            metric_name,
                            f"q{q}",
                            ".txt",
                        ]
                    )
                    os.makedirs(_path, exist_ok=True)
                    np.savetxt(
                        os.path.join(_path, _file_name), np.atleast_1d(np.asarray(metric_val))
                    )

            # save extra result
            final_data = mo_result.try_get_final_datasets()[OBJECTIVE]
            query, obs = final_data.query_points, final_data.observations
            oracle_data = mo_result.try_get_pending_oracle()
            if oracle_data is not None:
                x_oracle = tf.concat(
                    [
                        mo_result.history[0].datasets[OBJECTIVE].query_points,
                        tf.convert_to_tensor(oracle_data),
                    ],
                    axis=0,
                )
                _path = os.path.join(
                    path_prefix, kwargs.get("acq_name"), kwargs["file_info_prefix"]
                )
                os.makedirs(_path, exist_ok=True)
                _file_name = "_".join(
                    [
                        kwargs["benchmark_name"],
                        file_identifier_str,
                        "X_Oracle.txt" f"q{q}",
                        ".txt",
                    ]
                )
                np.savetxt(
                    os.path.join(_path, _file_name),
                    x_oracle,
                )

            # save all sampled observations
            _path = os.path.join(path_prefix, kwargs.get("acq_name"), kwargs["file_info_prefix"])
            _file_name = "_".join([kwargs["benchmark_name"], file_identifier_str, "X.txt"])
            os.makedirs(_path, exist_ok=True)
            np.savetxt(os.path.join(_path, _file_name), query.numpy())

            _path = os.path.join(path_prefix, kwargs.get("acq_name"), kwargs["file_info_prefix"])
            os.makedirs(_path, exist_ok=True)
            _file_name = "_".join([kwargs["benchmark_name"], file_identifier_str, "obs.txt"])
            np.savetxt(os.path.join(_path, _file_name), obs.numpy())

        if is_return:
            return mo_result
    except Exception as e:
        print(e)
        print(f"Exp {file_identifier_str} failed, skip")
        return


def ff_iu_parallel_benchmarker(workers: int, exp_repeat: int, **kwargs):
    """
    Parallel robust Bayesian Optimization benchmarker
    """
    from multiprocessing import Pool, set_start_method

    try:
        set_start_method("spawn")
    except:
        pass

    import numpy as np
    import tensorflow as tf

    tf.debugging.assert_greater_equal(workers, 1)
    tf.debugging.assert_greater_equal(exp_repeat, 1)

    if workers == 1:  # serial
        for i in range(exp_repeat):
            ff_iu_serial_benchmarker(i, **kwargs)
    else:  # parallel
        workers = exp_repeat if workers > exp_repeat else workers
        from functools import partial

        pb = partial(ff_iu_serial_benchmarker, **kwargs, is_return=False)
        for parallel_work in np.arange(0, exp_repeat, workers):
            with Pool(workers) as p:
                _ = p.map_async(pb, np.arange(parallel_work, parallel_work + workers)).get()
