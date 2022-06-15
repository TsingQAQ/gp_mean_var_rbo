import os

import tensorflow as tf

from docs.exp.FF_Variance.robust_bayesian_optimization_exp.var_as_con_acq_exp.interface_var_con import (
    main,
)


# FF-MV-qECI
def run_QFF_MV_qECI_normal():
    print("Run FF-MV-qECI")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=40,
        n_tp="normal",
        n_p=[[0.005]],
        q=1,
        fp=os.path.join("normal", "q1_qff"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.005]],
            "ff_method": "QFF",
            "opt_ff_num": 128,
            "infer_mc_num": 10000,
            "variance_threshold": 1.0,
            "mc_num": 128,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 10.794858459040201,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Forrester_MV_Normal_0.005_Opt_X.txt",
        kw_metrics={"worst_obj_val": 10.794858459040201},
        vc=1.0,
        which_rec="acquisition.recommend",
    )


# Random
def run_Random_normal():
    print("Run Random")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="Random",
        n=40,
        n_tp="normal",
        n_p=[[0.005]],
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.005]],
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Forrester_MV_Normal_0.005_Opt_X.txt",
        kw_metrics={"worst_obj_val": 10.794858459040201},
        vc=1.0,
    )


# Uncertainty Sampling [iwazaki2021mean]
def run_US_normal():
    print("Run US")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="US",
        n=40,
        n_tp="normal",
        n_p=[[0.005]],
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.005]],
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.005], tf.float64)),
            "kw_al": {"which_al_obj": "std"},
        },
        ref_optf="Forrester_MV_Normal_0.005_Opt_X.txt",
        kw_metrics={"worst_obj_val": 10.794858459040201},
        vc=1.0,
    )


def run_CO_MVA_BO_normal():
    print("Run CO-MVA-BO")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="CO-MVA-BO",
        n=40,
        n_tp="normal",
        n_p=[[0.005]],
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.005]],
            "approx_mc_num": 100,
            "tau": 0.001,
            "variance_threshold": 1.0,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.005], tf.float64)),
            "kw_al": {"which_al_obj": "P_std"},
        },
        ref_optf="Forrester_MV_Normal_0.005_Opt_X.txt",
        vc=1.0,
        kw_metrics={"worst_obj_val": 10.794858459040201},
        which_rec="acquisition.recommend",
    )


# RFF-MV-qECI
def run_RFF_MV_qECI_normal():
    print("Run FF-MV-qECI")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=40,
        n_tp="normal",
        n_p=[[0.005]],
        q=1,
        fp=os.path.join("normal", "q1_rff"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.005]],
            "ff_method": "RFF",
            "opt_ff_num": 128,
            "infer_mc_num": 10000,
            "variance_threshold": 1.0,
            "mc_num": 128,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 10.794858459040201,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Forrester_MV_Normal_0.005_Opt_X.txt",
        kw_metrics={"worst_obj_val": 10.794858459040201},
        vc=1.0,
        which_rec="acquisition.recommend",
    )


# FF-MV-qECI
def run_QFF_MV_qECI_uniform():
    print("Run FF-MV-qECI")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=40,
        n_tp="uniform",
        n_p=[0.1],
        q=1,
        fp=os.path.join("uniform", "q1_qff"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.1],
            "ff_method": "QFF",
            "opt_ff_num": 128,
            "infer_mc_num": 10000,
            "variance_threshold": 1.0,
            "mc_num": 128,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 12.136671074895611,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Forrester_MV_Uniform_0.1_Opt_F.txt",
        kw_metrics={"worst_obj_val": 12.136671074895611},
        vc=1.0,
        which_rec="acquisition.recommend",
    )


# Random
def run_Random_uniform():
    print("Run Random")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="Random",
        n=40,
        n_tp="uniform",
        n_p=[0.1],
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.1],
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Forrester_MV_Uniform_0.1_Opt_F.txt",
        kw_metrics={"worst_obj_val": 12.136671074895611},
        vc=1.0,
    )


# Uncertainty Sampling [iwazaki2021mean]
def run_US_uniform():
    print("Run US")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="US",
        n=40,
        n_tp="uniform",
        n_p=[0.1],
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.1],
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={"x_delta": tf.constant([0.1], tf.float64), "kw_al": {"which_al_obj": "std"}},
        ref_optf="Forrester_MV_Uniform_0.1_Opt_F.txt",
        kw_metrics={"worst_obj_val": 12.136671074895611},
        vc=1.0,
    )


def run_CO_MVA_BO_uniform():
    print("Run CO-MVA-BO")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="CO-MVA-BO",
        n=40,
        n_tp="uniform",
        n_p=[0.1],
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.1],
            "approx_mc_num": 100,
            "tau": 0.001,
            "variance_threshold": 1.0,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": tf.constant([0.1], tf.float64),
            "kw_al": {"which_al_obj": "P_std"},
        },
        ref_optf="Forrester_MV_Uniform_0.1_Opt_F.txt",
        vc=1.0,
        kw_metrics={"worst_obj_val": 12.136671074895611},
        which_rec="acquisition.recommend",
    )


# RFF-MV-qECI
def run_RFF_MV_qECI_uniform():
    print("Run FF-MV-qECI")
    main(
        cfg="forrester",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=40,
        n_tp="uniform",
        n_p=[0.1],
        q=1,
        fp=os.path.join("uniform", "q1_rff"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.1],
            "ff_method": "RFF",
            "opt_ff_num": 128,
            "infer_mc_num": 10000,
            "variance_threshold": 1.0,
            "mc_num": 128,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 12.136671074895611,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Forrester_MV_Uniform_0.1_Opt_F.txt",
        kw_metrics={"worst_obj_val": 12.136671074895611},
        vc=1.0,
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
