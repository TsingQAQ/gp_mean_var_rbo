import tensorflow as tf
from tensorflow_probability import distributions as tfd

from trieste.objectives.single_objectives import (
    Bird,
    Branin,
    Conceptual_Low_Drag_Wing_Design,
    Forrester,
    Hartmann_3,
    Robot_Pushing_3D,
    SinLinear,
)
from trieste.utils.metrics import AverageHausdoff, log_HV_differenceV2
from trieste.utils.post_recommender import mean_var_pareto_front_on_mean_based_on_data

sinlinear = {
    "pb": SinLinear,
    "pb_name": "SinLinear",
    "recommender": {"In_Sample": mean_var_pareto_front_on_mean_based_on_data},
    "post_metrics": {"Log_Hv": log_HV_differenceV2},
    "robustness_MC_num": 10000,
    "kwargs_for_objective": {},
    "kwargs_for_kernel": {
        "lengthscales_prior": tfd.LogNormal(
            tf.math.log(tf.constant([1.0], dtype=tf.float64)),
            tf.constant([0.5], dtype=tf.float64),
        )
    },
}


forrester = {
    "pb": Forrester,
    "pb_name": "Forrester",
    "recommender": {"In_Sample": mean_var_pareto_front_on_mean_based_on_data},
    "post_metrics": {"Log_Hv": log_HV_differenceV2},
    "robustness_MC_num": 10000,
    "kwargs_for_objective": {},
    "kwargs_for_kernel": {
        "lengthscales_prior": tfd.LogNormal(
            tf.math.log(tf.constant([1.0], dtype=tf.float64)),
            tf.constant([0.5], dtype=tf.float64),
        )
    },
}


branin = {
    "pb": Branin,
    "pb_name": "Branin",
    "recommender": {"In_Sample": mean_var_pareto_front_on_mean_based_on_data},
    "post_metrics": {"Log_Hv": log_HV_differenceV2},
    "robustness_MC_num": 10000,
    "kwargs_for_objective": {},
    "kwargs_for_kernel": {
        "lengthscales_prior": tfd.LogNormal(
            tf.math.log(tf.constant([1.0] * 2, dtype=tf.float64)),
            tf.constant([0.5], dtype=tf.float64),
        )
    },
}


bird = {
    "pb": Bird,
    "pb_name": "Bird",
    "recommender": {"In_Sample": mean_var_pareto_front_on_mean_based_on_data},
    "post_metrics": {"AVD": AverageHausdoff, "Log_Hv": log_HV_differenceV2},
    "robustness_MC_num": 10000,
    "kwargs_for_objective": {},
    "kwargs_for_kernel": {
        "lengthscales_prior": tfd.LogNormal(
            tf.math.log(tf.constant([1.0] * 2, dtype=tf.float64)),
            tf.constant([0.5], dtype=tf.float64),
        )
    },
}

conceptual_low_drag_wing_design = {
    "pb": Conceptual_Low_Drag_Wing_Design,
    "pb_name": "Conceptual_Low_Drag_Wing_Design",
    "recommender": {"In_Sample": mean_var_pareto_front_on_mean_based_on_data},
    "post_metrics": {"Log_Hv": log_HV_differenceV2},
    "robustness_MC_num": 500,
    "kwargs_for_objective": {"weight": 100.0, "velocity": 20.0},
    "kwargs_for_kernel": {
        "lengthscales_prior": tfd.LogNormal(
            tf.math.log(tf.constant([1.0] * 2, dtype=tf.float64)),
            tf.constant([0.5], dtype=tf.float64),
        )
    },
}


hartmann3 = {
    "pb": Hartmann_3,
    "pb_name": "Hartmann_3",
    "recommender": {"In_Sample": mean_var_pareto_front_on_mean_based_on_data},
    "post_metrics": {"Log_Hv": log_HV_differenceV2},
    "robustness_MC_num": 10000,
    "kwargs_for_objective": {},
    "kwargs_for_kernel": {
        "lengthscales_prior": tfd.LogNormal(
            tf.math.log(tf.constant([1.0] * 3, dtype=tf.float64)),
            tf.constant([0.5], dtype=tf.float64),
        )
    },
}
