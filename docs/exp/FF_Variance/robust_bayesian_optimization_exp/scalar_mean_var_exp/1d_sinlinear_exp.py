import os

import tensorflow as tf

from docs.exp.FF_Variance.robust_bayesian_optimization_exp.scalar_mean_var_exp.interface_scalar_mean_var import (
    main,
)


def run_QFF_SMV_qEI_normal():
    print("Run FF-SMV-qEI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-SMV-qEI",
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
            "alpha_var": 2.0,
            "mc_num": 128,
            "implicit_sample": True,
        },
        ref_optf="SinLinear_MV_Normal_0.001_Opt_F.txt",
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        sc_alpha=2.0,
    )


# Random
def run_Random_normal():
    print("Run Random")
    # Random
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
        sc_alpha=2.0,
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
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.001], tf.float64)),
            "kw_al": {"which_al_obj": "std"},
        },
        kw_optimize={"acquire_return_builder": True},
        ref_optf="SinLinear_MV_Normal_0.001_Opt_F.txt",
        sc_alpha=2.0,
    )


def run_MT_MVA_BO_normal():
    print("Run CO_MVA_BO")
    # MT-MVA-BO [iwazaki2021mean]
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="MT-MVA-BO",
        n=40,
        n_tp="normal",
        n_p=[[0.001]],
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={"noise_type": "normal", "noise_param": [[0.001]], "approx_mc_num": 100, "alpha": 2},
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.001], tf.float64)),
            "kw_al": {"which_al_obj": "P_std"},
        },
        ref_optf="SinLinear_MV_Normal_0.001_Opt_F.txt",
        sc_alpha=2.0,
        which_rec="acquisition.recommend",
    )


def run_RFF_SMV_qEI_normal():
    print("Run FF-SMV-qEI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-SMV-qEI",
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
            "alpha_var": 2.0,
            "mc_num": 128,
            "implicit_sample": True,
        },
        ref_optf="SinLinear_MV_Normal_0.001_Opt_F.txt",
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        sc_alpha=2.0,
    )


def run_QFF_SMV_qEI_uniform():
    print("Run FF-SMV-qEI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-SMV-qEI",
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
            "alpha_var": 2.0,
            "mc_num": 128,
            "implicit_sample": True,
        },
        ref_optf="SinLinear_MV_Uniform_0.1_Opt_F.txt",
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        sc_alpha=2.0,
    )


# Random
def run_Random_uniform():
    print("Run Random")
    # Random
    main(
        cfg="sinlinear",
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
        ref_optf="SinLinear_MV_Uniform_0.1_Opt_F.txt",
        sc_alpha=2.0,
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
        kw_optimizer={"x_delta": tf.constant([0.1], tf.float64), "kw_al": {"which_al_obj": "std"}},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="SinLinear_MV_Uniform_0.1_Opt_F.txt",
        sc_alpha=2.0,
    )


def run_MT_MVA_BO_uniform():
    print("Run CO_MVA_BO")
    # MT-MVA-BO [iwazaki2021mean]
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="MT-MVA-BO",
        n=40,
        n_tp="uniform",
        n_p=[0.1],
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={"noise_type": "uniform", "noise_param": [0.1], "approx_mc_num": 100, "alpha": 2},
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimizer={
            "x_delta": tf.constant([0.1], tf.float64),
            "kw_al": {"which_al_obj": "P_std"},
        },
        ref_optf="SinLinear_MV_Uniform_0.1_Opt_F.txt",
        sc_alpha=2.0,
        which_rec="acquisition.recommend",
    )


def run_RFF_SMV_qEI_uniform():
    print("Run FF-SMV-qEI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-SMV-qEI",
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
            "alpha_var": 2.0,
            "mc_num": 128,
            "implicit_sample": True,
        },
        ref_optf="SinLinear_MV_Uniform_0.1_Opt_F.txt",
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        sc_alpha=2.0,
    )


exp_cfg = {
    1: run_QFF_SMV_qEI_normal,
    2: run_Random_normal,
    3: run_US_normal,
    4: run_MT_MVA_BO_normal,
    5: run_RFF_SMV_qEI_normal,
    6: run_QFF_SMV_qEI_uniform,
    7: run_Random_uniform,
    8: run_US_uniform,
    9: run_MT_MVA_BO_uniform,
    10: run_RFF_SMV_qEI_uniform,
}


if __name__ == "__main__":
    which_to_run = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for which in which_to_run:
        exp_cfg[which]()
