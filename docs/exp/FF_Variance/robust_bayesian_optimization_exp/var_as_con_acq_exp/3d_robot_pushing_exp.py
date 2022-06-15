import os

import tensorflow as tf

from docs.exp.FF_Variance.robust_bayesian_optimization_exp.var_as_con_acq_exp.interface_var_con import (
    main,
)

"""
It has been checked, that both uniform and normal's worst mean is very similar
"""

# 4, 3, normal worst 2.828408177357718 VC 0.03
# FF-MV-qECI
def run_QFF_MV_qECI_normal():
    print("Run FF-MV-qECI")
    main(
        cfg="robot_pushing3",
        r=30,
        c=10,
        acq="FF-MV-qECI",
        n=100,
        n_tp="normal",
        n_p=tf.constant(
            [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
            dtype=tf.float64,
        ),
        q=1,
        fp=os.path.join("normal", "q1_qff"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
                dtype=tf.float64,
            ),
            "ff_method": "QFF",
            "opt_ff_num": 12,
            "infer_mc_num": 10000,
            "variance_threshold": 0.05,
            "mc_num": 128,
            "max_batch_element": 20,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 4.253299581555762,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.253299581555762},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Normal_Opt_F.txt",
        vc=0.05,
        which_rec="acquisition.recommend",
    )


# Random
def run_Random_normal():
    print("Run Random")
    main(
        cfg="robot_pushing3",
        r=30,
        c=15,
        acq="Random",
        n=100,
        n_tp="normal",
        n_p=tf.constant(
            [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
            dtype=tf.float64,
        ),
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
                dtype=tf.float64,
            ),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.253299581555762},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Normal_Opt_F.txt",
        vc=0.05,
    )


# Uncertainty Sampling [iwazaki2021mean]
def run_US_normal():
    print("Run US")
    main(
        cfg="robot_pushing3",
        r=30,
        c=15,
        acq="US",
        n=100,
        n_tp="normal",
        n_p=tf.constant(
            [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
            dtype=tf.float64,
        ),
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
                dtype=tf.float64,
            ),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.253299581555762},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimizer={
            "x_delta": 1.96
            * tf.sqrt(tf.constant([0.0004, 0.0004, 7.13436385255648e-05], tf.float64)),
            "kw_al": {"which_al_obj": "std"},
        },
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Normal_Opt_F.txt",
        vc=0.05,
    )


def run_CO_MVA_BO_normal():
    print("Run CO_MVA_BO")
    # CO-MVA-BO [iwazaki2021mean]
    main(
        cfg="robot_pushing3",
        r=30,
        c=15,
        acq="CO-MVA-BO",
        n=100,
        n_tp="normal",
        n_p=tf.constant(
            [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
            dtype=tf.float64,
        ),
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
                dtype=tf.float64,
            ),
            "approx_mc_num": 100,
            "tau": 0.05,
            "variance_threshold": 0.05,
        },
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.253299581555762},
        kw_optimizer={
            "x_delta": 1.96
            * tf.sqrt(tf.constant([0.0004, 0.0004, 7.13436385255648e-05], tf.float64)),
            "kw_al": {"which_al_obj": "P_std"},
        },
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Normal_Opt_F.txt",
        vc=0.05,
    )


# FF-MV-qECI
def run_RFF_MV_qECI_normal():
    print("Run FF-MV-qECI")
    main(
        cfg="robot_pushing3",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=100,
        n_tp="normal",
        n_p=tf.constant(
            [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
            dtype=tf.float64,
        ),
        q=1,
        fp=os.path.join("normal", "q1_rff"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]],
                dtype=tf.float64,
            ),
            "ff_method": "RFF",
            "opt_ff_num": 1728,
            "infer_mc_num": 10000,
            "variance_threshold": 0.05,
            "mc_num": 128,
            "max_batch_element": 20,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 4.253299581555762,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.253299581555762},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Normal_Opt_F.txt",
        vc=0.05,
        which_rec="acquisition.recommend",
    )


# FF-MV-qECI
def run_QFF_MV_qECI_uniform():
    print("Run FF-MV-qECI")
    main(
        cfg="robot_pushing3",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=100,
        n_tp="uniform",
        n_p=tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1_qff"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
            "ff_method": "QFF",
            "opt_ff_num": 12,
            "infer_mc_num": 10000,
            "variance_threshold": 0.03 ,
            "mc_num": 128,
            "max_batch_element": 20,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 4.2693942633336,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.2693942633336},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Uniform_Opt_F.txt",
        vc=0.03 ,
        which_rec="acquisition.recommend",
    )


# Random
def run_Random_uniform():
    print("Run Random")
    main(
        cfg="robot_pushing3",
        r=30,
        c=15,
        acq="Random",
        n=100,
        n_tp="uniform",
        n_p=tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.2693942633336},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Uniform_Opt_F.txt",
        vc=0.03,
    )


# Uncertainty Sampling [iwazaki2021mean]
def run_US_uniform():
    print("Run US")
    main(
        cfg="robot_pushing3",
        r=30,
        c=15,
        acq="US",
        n=100,
        n_tp="uniform",
        n_p=tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.2693942633336},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimizer={
            "x_delta": tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
            "kw_al": {"which_al_obj": "std"},
        },
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Uniform_Opt_F.txt",
        vc=0.03 ,
    )


def run_CO_MVA_BO_uniform():
    print("Run CO_MVA_BO")
    # CO-MVA-BO [iwazaki2021mean]
    main(
        cfg="robot_pushing3",
        r=30,
        c=15,
        acq="CO-MVA-BO",
        n=100,
        n_tp="uniform",
        n_p=tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
            "approx_mc_num": 100,
            "tau": 0.05,
            "variance_threshold": 0.03 ,
        },
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.2693942633336},
        kw_optimizer={
            "x_delta": tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
            "kw_al": {"which_al_obj": "P_std"},
        },
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Uniform_Opt_F.txt",
        vc=0.03,
    )


# FF-MV-qECI
def run_RFF_MV_qECI_uniform():
    print("Run FF-MV-qECI")
    main(
        cfg="robot_pushing3",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=100,
        n_tp="uniform",
        n_p=tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1_rff"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.03, 0.03, 0.01], dtype=tf.float64),
            "ff_method": "RFF",
            "opt_ff_num": 1728,
            "infer_mc_num": 10000,
            "variance_threshold": 0.03 ,
            "mc_num": 128,
            "max_batch_element": 20,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 4.2693942633336,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 4.2693942633336},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Robot_Pushing_3D_Uniform_Opt_F.txt",
        vc=0.03 ,
        which_rec="acquisition.recommend",
    )


exp_cfg = {
    1: run_QFF_MV_qECI_normal,
    2: run_Random_normal,
    3: run_US_normal,
    4: run_CO_MVA_BO_normal,
    5: run_RFF_MV_qECI_normal,
    6: run_QFF_MV_qECI_uniform,
    7: run_Random_uniform,
    8: run_US_uniform,
    9: run_CO_MVA_BO_uniform,
    10: run_RFF_MV_qECI_uniform,
}

if __name__ == "__main__":
    which_to_run = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for which in which_to_run:
        exp_cfg[which]()
