import tensorflow as tf
from tensorflow_probability import distributions as tfd

from trieste.objectives.single_objectives import (
    Bird,
    Branin,
    Forrester,
    Hartmann_3,
    Robot_Pushing_3D,
    SinLinear,
)
from trieste.utils.metrics import Scalarization_Infer_regret
from trieste.utils.post_recommender import mean_var_scalarization_on_mean_based_on_data

sinlinear = {
    "pb": SinLinear,
    "pb_name": "SinLinear",
    "recommender": {"In_Sample": mean_var_scalarization_on_mean_based_on_data},
    "post_metrics": {"Regret": Scalarization_Infer_regret},
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
    "recommender": {"In_Sample": mean_var_scalarization_on_mean_based_on_data},
    "post_metrics": {"Regret": Scalarization_Infer_regret},
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
    "recommender": {"In_Sample": mean_var_scalarization_on_mean_based_on_data},
    "post_metrics": {"Regret": Scalarization_Infer_regret},
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
    "recommender": {"In_Sample": mean_var_scalarization_on_mean_based_on_data},
    "post_metrics": {"Regret": Scalarization_Infer_regret},
    "robustness_MC_num": 10000,
    "kwargs_for_objective": {},
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
    "recommender": {"In_Sample": mean_var_scalarization_on_mean_based_on_data},
    "post_metrics": {"Regret": Scalarization_Infer_regret},
    "robustness_MC_num": 10000,
    "kwargs_for_objective": {},
    "kwargs_for_kernel": {
        "lengthscales_prior": tfd.LogNormal(
            tf.math.log(tf.constant([1.0] * 3, dtype=tf.float64)),
            tf.constant([0.5], dtype=tf.float64),
        )
    },
}

robot_pushing3 = {
    "pb": Robot_Pushing_3D,
    "pb_name": "Robot_Pushing_3D",
    "recommender": {"In_Sample": mean_var_scalarization_on_mean_based_on_data},
    "post_metrics": {"Regret": Scalarization_Infer_regret},
    "robustness_MC_num": 1000,
    "kwargs_for_objective": {"obj_loc_x": 4.0, "obj_loc_y": 3.0},
    "kwargs_for_kernel": {
        "lengthscales_prior": tfd.LogNormal(
            tf.math.log(tf.constant([1.0] * 3, dtype=tf.float64)),
            tf.constant([0.5], dtype=tf.float64),
        )
    },
}
