import os

import tensorflow as tf

from docs.exp.FF_Variance.robust_bayesian_optimization_exp.mo_mean_var_exp.interface_mean_var import (
    main,
)


# QFF-MV-qEHVI
def run_QFF_MV_qEHVI_normal():
    print("Run FF-MV-qEHVI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-MV-qEHVI",
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
            "mc_num": 128,
            "implicit_sample": True
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="SinLinear_MV_Normal_0.01_PF.txt",
        kw_metrics={
            "Log_Hv": {"ref_point": [0.03003507, 0.07371671]},
            "AVD": {"scaler": ([-1.1416561e00, 6.7403522e-04], [-0.01521673, 0.06952376])},
        },
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
        ref_pf="SinLinear_MV_Normal_0.01_PF.txt",
        kw_metrics={
            "Log_Hv": {"ref_point": [0.03003507, 0.07371671]},
            "AVD": {"scaler": ([-1.1416561e00, 6.7403522e-04], [-0.01521673, 0.06952376])},
        },
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
        ref_pf="SinLinear_MV_Normal_0.01_PF.txt",
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.001], tf.float64)),
            "kw_al": {"which_al_obj": "std"},
        },
        kw_metrics={
            "Log_Hv": {"ref_point": [0.03003507, 0.07371671]},
            "AVD": {"scaler": ([-1.1416561e00, 6.7403522e-04], [-0.01521673, 0.06952376])},
        },
    )


def run_MO_MVA_BO_normal():
    print("Run MO_MVA_BO")
    # MO-MVA-BO [iwazaki2021mean]
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="MO-MVA-BO",
        n=40,
        n_tp="normal",
        n_p=[[0.001]],
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": [[0.01]],
            "approx_mc_num": 100,
            "tau": 0.001,
            "beta_t": 2.0,
            "max_batch_element_num": 500,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.001], tf.float64)),
            "kw_al": {"which_al_obj": "P_std"},
        },
        ref_pf="SinLinear_MV_Normal_0.01_PF.txt",
        kw_metrics={
            "Log_Hv": {"ref_point": [0.03003507, 0.07371671]},
            "AVD": {"scaler": ([-1.1416561e00, 6.7403522e-04], [-0.01521673, 0.06952376])},
        },
        which_rec="acquisition.recommend",
    )


# RFF-MV-qEHVI
def run_RFF_MV_qEHVI_normal():
    print("Run FF-MV-qEHVI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-MV-qEHVI",
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
            "mc_num": 128,
            "implicit_sample": True
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="SinLinear_MV_Normal_0.01_PF.txt",
        kw_metrics={
            "Log_Hv": {"ref_point": [0.03003507, 0.07371671]},
            "AVD": {"scaler": ([-1.1416561e00, 6.7403522e-04], [-0.01521673, 0.06952376])},
        },
    )


# QFF-MV-qEHVI
def run_QFF_MV_qEHVI_uniform():
    print("Run FF-MV-qEHVI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-MV-qEHVI",
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
            "mc_num": 128,
            "implicit_sample": True
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="SinLinear_MV_Uniform_0.05_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.03320457, 0.02952188]}},
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
        ref_pf="SinLinear_MV_Uniform_0.05_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.03320457, 0.02952188]}},
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
        ref_pf="SinLinear_MV_Uniform_0.05_PF.txt",
        kw_optimizer={"x_delta": tf.constant([0.05], tf.float64), "kw_al": {"which_al_obj": "std"}},
        kw_metrics={"Log_Hv": {"ref_point": [0.03320457, 0.02952188]}},
    )


#
def run_MO_MVA_BO_uniform():
    print("Run MO_MVA_BO")
    # MO-MVA-BO [iwazaki2021mean]
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="MO-MVA-BO",
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
            "beta_t": 2.0,
            "max_batch_element_num": 500,
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": tf.constant([0.05], tf.float64),
            "kw_al": {"which_al_obj": "P_std"},
        },
        ref_pf="SinLinear_MV_Uniform_0.05_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.03320457, 0.02952188]}},
        which_rec="acquisition.recommend",
    )


# RFF-MV-qEHVI
def run_RFF_MV_qEHVI_uniform():
    print("Run FF-MV-qEHVI")
    main(
        cfg="sinlinear",
        r=30,
        c=15,
        acq="FF-MV-qEHVI",
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
            "mc_num": 128,
            "implicit_sample": True
        },
        kw_bench={"doe_num": 5, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 2},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="SinLinear_MV_Uniform_0.05_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.03320457, 0.02952188]}},
    )


exp_cfg = {
    1: run_QFF_MV_qEHVI_normal,
    2: run_Random_normal,
    3: run_US_normal,
    4: run_MO_MVA_BO_normal,
    5: run_RFF_MV_qEHVI_normal,
    6: run_QFF_MV_qEHVI_uniform,
    7: run_Random_uniform,
    8: run_US_uniform,
    9: run_MO_MVA_BO_uniform,
    10: run_RFF_MV_qEHVI_uniform,
}

if __name__ == "__main__":
    which_to_run = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for which in which_to_run:
        exp_cfg[which]()
