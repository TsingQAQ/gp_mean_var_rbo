import argparse
import json
import os
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from docs.exp.FF_Variance.robust_bayesian_optimization_exp.robust_bayesian_optimizer import (
    ff_iu_parallel_benchmarker,
)
from docs.exp.FF_Variance.robust_bayesian_optimization_exp.utils.extracting_results import (
    get_performance_metric_from_robust_bayesian_optimize_result,
)
from docs.exp.FF_Variance.robust_bayesian_optimization_exp.var_as_con_acq_exp.cfg import pb_cfgs
from trieste.acquisition.function.robust import CO_MVA_BO, US, FF_MV_qECI, Random_Acq
from trieste.space import Box


def main(**kwargs):
    if kwargs != {}:
        cfg_name = kwargs["cfg"]
        exp_repeat = kwargs["r"]
        workers = kwargs["c"]
        acq_name = kwargs["acq"]
        total_iter = kwargs["n"]
        noise_type = kwargs["n_tp"]
        noise_param = kwargs["n_p"]
        file_info_prefix = kwargs["file_info_prefix"] if "file_info_prefix" in kwargs else ""
        q = kwargs["q"] if "q" in kwargs else 1
        kwargs_for_acq = kwargs["kw_acq"] if "kw_acq" in kwargs else {}
        kwargs_for_benchmarker = kwargs["kw_bench"] if "kw_bench" in kwargs else {}
        kwargs_for_acquisition_rule = kwargs["kw_rule"] if "kw_rule" in kwargs else {}
        kwargs_for_optimizer = kwargs["kw_optimizer"] if "kw_optimizer" in kwargs else {}
        kwargs_for_optimize = kwargs["kw_optimize"] if "kw_optimize" in kwargs else {}
        ref_pf_filename = kwargs["ref_optf"] if "ref_optf" in kwargs else {}
        kw_metrics = kwargs["kw_metrics"] if "kw_metrics" in kwargs else {}
        file_info_prefix = kwargs["fp"] if "fp" in kwargs else {}
        variance_threshold = kwargs["vc"] if "vc" in kwargs else ValueError
        which_model_for_recommender = kwargs["which_rec"] if "which_rec" in kwargs else "mean_var"
    else:  # use argparse
        parser = argparse.ArgumentParser()

        parser.add_argument("-cfg")
        parser.add_argument("-n", "--total_iter", type=int)
        parser.add_argument("-n_tp", "--noise_type", type=str)
        parser.add_argument("-n_p", "--noise_param")
        parser.add_argument("-c", "--core", type=int)
        parser.add_argument("-r", "--repeat", type=int)
        parser.add_argument("-vc", "--variance_threshold", type=float)
        parser.add_argument("-acq")
        parser.add_argument("-ref_optf", "--ref_pf_filename")
        parser.add_argument("-which_rec", default="mean_var")
        parser.add_argument("-kw_acq", type=json.loads)
        parser.add_argument("-kw_bench", type=json.loads)
        parser.add_argument("-kw_rule", type=json.loads)
        parser.add_argument("-kw_optimizer", type=json.loads)
        parser.add_argument("-kw_optimize", type=json.loads)
        parser.add_argument("-kw_metrics", type=json.loads)

        # optional args
        parser.add_argument("-fp", "--file_info_prefix", default="", type=str)
        parser.add_argument("-q", "--batch_query", default=1, type=int)

        _args = parser.parse_args()
        cfg_name = _args.cfg
        exp_repeat = _args.repeat
        noise_type = _args.noise_type
        noise_param = _args.noise_param
        variance_threshold = _args.variance_threshold
        workers = _args.core
        acq_name = _args.acq
        total_iter = _args.total_iter
        ref_pf_filename = _args.ref_pf_filename
        kwargs_for_benchmarker = _args.kw_bench
        kwargs_for_acquisition_rule = _args.kw_rule
        kwargs_for_optimizer = _args.kw_optimizer
        kwargs_for_optimize = _args.kw_optimize
        kw_metrics = _args.kw_metrics
        which_model_for_recommender = _args.which_rec

        file_info_prefix = _args.file_info_prefix if _args.file_info_prefix else ""
        q = _args.batch_query if _args.batch_query else 1
        kwargs_for_acq = _args.kw_acq if _args.kw_acq else {}
    try:
        pb_cfg = getattr(pb_cfgs, cfg_name)
        pb_name = pb_cfg["pb_name"]
    except:
        raise NotImplementedError(
            rf"NotImplemented Problem: {pb_name} specified, however, it doesn\'t mean this "
            r"benchmark cannot be used for a new problem, in order to do so,"
            r"you may need to first write your own problem cfg in cfg/pb_cfgs.py"
        )

    noise_param = tf.convert_to_tensor(noise_param, dtype=tf.float64)
    if noise_type == "normal":
        pb_cfg["seed_sampler"] = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(shape=tf.shape(tf.convert_to_tensor(noise_param))[0], dtype=tf.float64),
            covariance_matrix=noise_param,
        )
    elif noise_type == "uniform":
        pb_cfg["seed_sampler"] = tfd.Uniform(-noise_param, noise_param)
    else:
        raise NotImplementedError

    # pb_cfg['optimum_x'] = NotImplementedError

    # Maybe can stored on local before hand
    if acq_name == "FF-MV-qECI":
        acq = FF_MV_qECI
    elif acq_name == "CO-MVA-BO":
        acq = CO_MVA_BO
    elif acq_name == "US":
        acq = US
    elif acq_name == "Random":
        acq = Random_Acq
    else:
        raise NotImplementedError(rf"NotImplemented Acquisition: {acq_name} specified")

    pb = pb_cfg["pb"](**pb_cfg["kwargs_for_objective"])

    _true_constrained_optima = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cfg",
        "ref_opts",
        pb_cfg["pb_name"],
        ref_pf_filename,
    )

    for rec_key, recommender in pb_cfg["recommender"].items():
        if rec_key == "In_Sample":
            pb_cfg["recommender"][rec_key] = partial(
                recommender, variance_threshold=variance_threshold
            )

    metrics = {}
    for metric_key, metric_func in pb_cfg["post_metrics"].items():
        if metric_key == "Regret":
            metrics[metric_key] = partial(
                metric_func,
                true_optimum=np.loadtxt(_true_constrained_optima),
                variance_threshold=variance_threshold,
                worst_function_val_in_design_space=kw_metrics["worst_obj_val"],
            )
        else:
            raise NotImplementedError(rf"Post Metric {metric_key} don\'t understood")

    ff_iu_parallel_benchmarker(
        benchmark_name=pb_name,
        acq_name=acq_name,
        file_info_prefix=file_info_prefix,
        exp_repeat=exp_repeat,
        workers=workers,
        acq=acq,
        q=q,
        total_iter=total_iter,
        **pb_cfg,
        path_prefix=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "exp_res", pb_cfg["pb_name"]
        ),
        post_profiler=partial(
            get_performance_metric_from_robust_bayesian_optimize_result,
            recommenders=pb_cfg["recommender"],
            design_space=Box(*pb.bounds),
            metrics=metrics,
            true_func=pb.fmean_fvar_objective(
                tf.expand_dims(pb_cfg["seed_sampler"].sample(pb_cfg["robustness_MC_num"]), axis=-2)
            ),
            batch_size=q,
            which_model_for_recommender=which_model_for_recommender,
        ),
        kwargs_for_acq=kwargs_for_acq,
        kwargs_for_acquisition_rule=kwargs_for_acquisition_rule,
        kwargs_for_optimizer=kwargs_for_optimizer,
        kwargs_for_optimize=kwargs_for_optimize,
        **kwargs_for_benchmarker,
    )
