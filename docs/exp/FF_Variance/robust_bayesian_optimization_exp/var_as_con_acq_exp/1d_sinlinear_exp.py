import os

import tensorflow as tf

from docs.exp.FF_Variance.robust_bayesian_optimization_exp.var_as_con_acq_exp.interface_var_con import (
    main,
)


# FF-MV-qEHVI
def run_QFF_MV_qECI_normal():
    print("Run QFF-MV-qECI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=40,
        n_tp="normal",
        n_p=[[0.001]],
        q=1,
        fp=os.path.join("normal", "q1_qff"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.001]],
            "ff_method": "QFF",
            "opt_ff_num": 128,
            "infer_mc_num": 10000,
            "variance_threshold": 0.14,
            "mc_num": 128,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 0.5908988891963954,
        },
        ref_optf="SinLinear_MV_Normal_0.001_Opt_F.txt",
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        kw_metrics={"worst_obj_val": 0.5908988891963954},
        vc=0.14,
        which_rec="acquisition.recommend",
    )


# Random
def run_Random_normal():
    print("Run Random")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="Random",
        n=40,
        n_tp="normal",
        n_p=[[0.001]],
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.001]],
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="SinLinear_MV_Normal_0.001_Opt_F.txt",
        kw_metrics={"worst_obj_val": 0.5908988891963954},
        vc=0.14,
    )


def run_US_normal():
    print("Run US")
    # Uncertainty Sampling [iwazaki2021mean]
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="US",
        n=40,
        n_tp="normal",
        n_p=[[0.001]],
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.001]],
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.001], tf.float64)),
            "kw_al": {"which_al_obj": "std"},
        },
        ref_optf="SinLinear_MV_Normal_0.001_Opt_F.txt",
        kw_metrics={"worst_obj_val": 0.5908988891963954},
        vc=0.14,
    )


def run_CO_MVA_BO_normal():
    print("Run CO_MVA_BO")
    # CO-MVA-BO [iwazaki2021mean]
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="CO-MVA-BO",
        n=40,
        n_tp="normal",
        n_p=[[0.001]],
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.001]],
            "approx_mc_num": 100,
            "tau": 0.001,
            "variance_threshold": 0.14,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.001], tf.float64)),
            "kw_al": {"which_al_obj": "P_std"},
        },
        kw_optimize={"acquire_return_builder": True},
        ref_optf="SinLinear_MV_Normal_0.001_Opt_F.txt",
        kw_metrics={"worst_obj_val": 0.5908988891963954},
        vc=0.14,
        which_rec="acquisition.recommend",
    )


# FF-MV-qEHVI
def run_RFF_MV_qECI_normal():
    print("Run RFF-MV-qECI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=40,
        n_tp="normal",
        n_p=[[0.001]],
        q=1,
        fp=os.path.join("normal", "q1_rff"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.001]],
            "ff_method": "RFF",
            "opt_ff_num": 128,
            "infer_mc_num": 10000,
            "variance_threshold": 0.14,
            "mc_num": 128,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 0.5908988891963954,
        },
        ref_optf="SinLinear_MV_Normal_0.001_Opt_F.txt",
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        kw_metrics={"worst_obj_val": 0.5908988891963954},
        vc=0.14,
        which_rec="acquisition.recommend",
    )


# FF-MV-qEHVI
def run_QFF_MV_qECI_uniform():
    print("Run QFF-MV-qECI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=40,
        n_tp="uniform",
        n_p=[0.05],
        q=1,
        fp=os.path.join("uniform", "q1_qff"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.05],
            "ff_method": "QFF",
            "opt_ff_num": 128,
            "infer_mc_num": 10000,
            "variance_threshold": 0.14,
            "mc_num": 128,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 0.54349843,
        },
        ref_optf="SinLinear_MV_Uniform_0.05_Opt_F.txt",
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        kw_metrics={"worst_obj_val": 0.54349843},
        vc=0.14,
        which_rec="acquisition.recommend",
    )


# Random
def run_Random_uniform():
    print("Run Random")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="Random",
        n=40,
        n_tp="uniform",
        n_p=[0.05],
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.05],
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="SinLinear_MV_Uniform_0.05_Opt_F.txt",
        kw_metrics={"worst_obj_val": 0.54349843},
        vc=0.14,
    )


def run_US_uniform():
    print("Run US")
    # Uncertainty Sampling [iwazaki2021mean]
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="US",
        n=40,
        n_tp="uniform",
        n_p=[0.05],
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.05],
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={"x_delta": tf.constant([0.05], tf.float64), "kw_al": {"which_al_obj": "std"}},
        ref_optf="SinLinear_MV_Uniform_0.05_Opt_F.txt",
        kw_metrics={"worst_obj_val": 0.54349843},
        vc=0.14,
    )


def run_CO_MVA_BO_uniform():
    print("Run CO_MVA_BO")
    # CO-MVA-BO [iwazaki2021mean]
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="CO-MVA-BO",
        n=40,
        n_tp="uniform",
        n_p=[0.05],
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.05],
            "approx_mc_num": 100,
            "tau": 0.001,
            "variance_threshold": 0.14,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_optimizer={
            "x_delta": tf.constant([0.05], tf.float64),
            "kw_al": {"which_al_obj": "P_std"},
        },
        kw_optimize={"acquire_return_builder": True},
        ref_optf="SinLinear_MV_Uniform_0.05_Opt_F.txt",
        kw_metrics={"worst_obj_val": 0.54349843},
        vc=0.14,
        which_rec="acquisition.recommend",
    )


# FF-MV-qEHVI
def run_RFF_MV_qECI_uniform():
    print("Run RFF-MV-qECI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=40,
        n_tp="uniform",
        n_p=[0.05],
        q=1,
        fp=os.path.join("uniform", "q1_rff"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": [0.05],
            "ff_method": "RFF",
            "opt_ff_num": 128,
            "infer_mc_num": 10000,
            "variance_threshold": 0.14,
            "mc_num": 128,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 0.54349843,
        },
        ref_optf="SinLinear_MV_Uniform_0.05_Opt_F.txt",
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        kw_metrics={"worst_obj_val": 0.54349843},
        vc=0.14,
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
