from typing import Callable, Mapping, Optional

import tensorflow as tf

from trieste.observer import OBJECTIVE
from trieste.space import Box


def get_performance_metric_from_robust_bayesian_optimize_result(
    optimization_result,
    recommenders: Mapping[str, Callable],
    metrics: Mapping[str, Callable],
    design_space: Box,
    true_func: [Callable, None] = None,
    batch_size: int = 1,
    which_model_for_recommender: str = "model",
    acq_name: str = None,
    record_frequency: Optional[int] = None,
    **kwargs,
):
    """
    Extract metric history from optimization history
    :param recommenders
    :param batch_size: after how many batch_size the metrics are getting re-evaluated, set to 1 when non-batch bo is used
    :param true_func
    :param design_space
    :param optimization_result
    :param metrics: {"HV": hv_indicator, "IGD": IGD_indicator}
    :param which_model_for_recommender what to use to make optimal inference, by default use model

    """
    # prepare recommender
    recommender_history = {}
    for rec_key in recommenders.keys():
        recommender_history[rec_key] = {}
        for metric_key in metrics.keys():
            recommender_history[rec_key][metric_key] = []

    oracle_data = optimization_result.try_get_pending_oracle()
    # history is the beginning state of a bo iter, hence we need to use the following way to extract the proper history
    # opt_history is the same length as the bo_iter, but contains the result after each bo iter
    opt_history = optimization_result.history + [optimization_result.final_result.unwrap()]

    # extracting:
    for hist, idx in zip(opt_history, tf.range(len(opt_history))):
        if record_frequency is None or idx == 0 or idx % record_frequency == 0:
            if which_model_for_recommender == "model":
                model = hist.models
            elif which_model_for_recommender == "mean_var":
                model = lambda at: tf.concat(
                    [hist.acquisition_state.fmean_mean(at), hist.acquisition_state.fvar_mean(at)],
                    -1,
                )
            elif which_model_for_recommender == "acquisition.recommend":
                pass
            else:
                raise NotImplementedError
            # data preparing
            if oracle_data is None:  # no pending oracle
                all_candidates = hist.datasets[OBJECTIVE].query_points
            else:  # has pending oracle, build own data points
                # get initial points (not recorded by oracle)
                bo_start_data = opt_history[0].datasets[OBJECTIVE].query_points
                current_bo_iter = idx + 1
                all_candidates = tf.concat(
                    [
                        bo_start_data[:-1, ...],
                        tf.convert_to_tensor(oracle_data[: current_bo_iter * batch_size, ...]),
                    ],
                    axis=0,
                )
            # mask data that in design space
            in_space_mask = tf.logical_and(
                tf.reduce_all(all_candidates >= design_space.lower, -1),
                tf.reduce_all(all_candidates <= design_space.upper, -1),
            )
            in_space_candidate = all_candidates[in_space_mask]

            # recommending optimal design
            for recommender_key, recommender in recommenders.items():
                if which_model_for_recommender == "acquisition.recommend":
                    model = hist.models[OBJECTIVE]
                    acq_recommender = hist.acquisition_state.get_recommend
                    if acq_name in ["MO-MVA-BO", "CO-MVA-BO"]:
                        in_space_candidate = design_space.discretize_v2(kwargs["tau"]).points
                    _optimum_x = acq_recommender(model, in_space_candidate)
                else:
                    _optimum_x = recommender(model, in_space_candidate)
                for metric_key, metric in metrics.items():
                    if true_func is not None:
                        recommender_history[recommender_key][metric_key].append(
                            metric(_optimum_x, true_func)
                        )
                    else:
                        recommender_history[recommender_key][metric_key].append(metric(_optimum_x))

                # extracting final recommendations
                if idx == len(opt_history) - 1:  # reached the final recommendation
                    recommender_history[recommender_key]["recommend_input"] = _optimum_x

    return recommender_history
