import os

import tensorflow as tf

from docs.exp.FF_Variance.robust_bayesian_optimization_exp.mo_mean_var_exp.interface_mean_var import (
    main,
)


# FF-MV-qEHVI
def run_QFF_MV_qEHVI_normal():
    print("Run QFF-MV-qEHVI")
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="FF-MV-qEHVI",
        n=50,
        n_tp="normal",
        n_p=tf.constant([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64),
        q=1,
        fp=os.path.join("normal", "q1_qff"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64
            ),
            "ff_method": "QFF",
            "opt_ff_num": 10,
            "infer_mc_num": 10000,
            "mc_num": 128,
            "max_batch_element": 20,
            "implicit_sample": True
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="Hartmann3_MV_Normal_Ind_0.01_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.11863389, 0.46989925]}},
    )


# Random
def run_Random_normal():
    print("Run Random")
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="Random",
        n=50,
        n_tp="normal",
        n_p=tf.constant([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64),
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64
            ),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="Hartmann3_MV_Normal_Ind_0.01_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.11863389, 0.46989925]}},
    )


# Uncertainty Sampling [iwazaki2021mean]
def run_US_normal():
    print("Run US")
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="US",
        n=50,
        n_tp="normal",
        n_p=tf.constant([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64),
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64
            ),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.01, 0.01, 0.01], tf.float64)),
            "kw_al": {"which_al_obj": "std"},
        },
        ref_pf="Hartmann3_MV_Normal_Ind_0.01_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.11863389, 0.46989925]}},
    )


def run_MO_MVA_BO_normal():
    print("Run MO_MVA_BO")
    # MO-MVA-BO [iwazaki2021mean]
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="MO-MVA-BO",
        n=50,
        n_tp="normal",
        n_p=tf.constant([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64),
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64
            ),
            "approx_mc_num": 100,
            "tau": 0.05,
            "beta_t": 2.0,
            "max_batch_element_num": 50,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.01, 0.01, 0.01], tf.float64)),
            "kw_al": {"which_al_obj": "P_std"},
        },
        ref_pf="Hartmann3_MV_Normal_Ind_0.01_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.11863389, 0.46989925]}},
        which_rec="acquisition.recommend",
    )


# FF-MV-qEHVI
def run_RFF_MV_qEHVI_normal():
    print("Run RFF-MV-qEHVI")
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="FF-MV-qEHVI",
        n=50,
        n_tp="normal",
        n_p=tf.constant([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64),
        q=1,
        fp=os.path.join("normal", "q1_rff"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant(
                [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64
            ),
            "ff_method": "RFF",
            "opt_ff_num": 1000,
            "infer_mc_num": 10000,
            "mc_num": 128,
            "max_batch_element": 20,
            "implicit_sample": True
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="Hartmann3_MV_Normal_Ind_0.01_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.11863389, 0.46989925]}},
    )


# FF-MV-qEHVI
def run_QFF_MV_qEHVI_uniform():
    print("Run QFF-MV-qEHVI")
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="FF-MV-qEHVI",
        n=50,
        n_tp="uniform",
        n_p=tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1_qff"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
            "ff_method": "QFF",
            "opt_ff_num": 10,
            "infer_mc_num": 10000,
            "mc_num": 128,
            "max_batch_element": 20,
            "implicit_sample": True
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="Hartmann3_MV_Uniform_Ind_0.15_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.12358115, 0.21843887]}},
    )


# Random
def run_Random_uniform():
    print("Run Random")
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="Random",
        n=50,
        n_tp="uniform",
        n_p=tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="Hartmann3_MV_Uniform_Ind_0.15_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.12358115, 0.21843887]}},
    )


# Uncertainty Sampling [iwazaki2021mean]
def run_US_uniform():
    print("Run US")
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="US",
        n=50,
        n_tp="uniform",
        n_p=tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
            "kw_al": {"which_al_obj": "std"},
        },
        ref_pf="Hartmann3_MV_Uniform_Ind_0.15_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.12358115, 0.21843887]}},
    )


def run_MO_MVA_BO_uniform():
    print("Run MO_MVA_BO")
    # MO-MVA-BO [iwazaki2021mean]
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="MO-MVA-BO",
        n=50,
        n_tp="uniform",
        n_p=tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
            "approx_mc_num": 100,
            "tau": 0.05,
            "beta_t": 2.0,
            "max_batch_element_num": 50,
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
            "kw_al": {"which_al_obj": "P_std"},
        },
        ref_pf="Hartmann3_MV_Uniform_Ind_0.15_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.12358115, 0.21843887]}},
        which_rec="acquisition.recommend",
    )


# FF-MV-qEHVI
def run_RFF_MV_qEHVI_uniform():
    print("Run RFF-MV-qEHVI")
    main(
        cfg="hartmann3",
        r=30,
        c=15,
        acq="FF-MV-qEHVI",
        n=50,
        n_tp="uniform",
        n_p=tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
        q=1,
        fp=os.path.join("uniform", "q1_rff"),
        kw_acq={
            "noise_type": "uniform",
            "noise_param": tf.constant([0.15, 0.15, 0.15], dtype=tf.float64),
            "ff_method": "RFF",
            "opt_ff_num": 1000,
            "infer_mc_num": 10000,
            "mc_num": 128,
            "max_batch_element": 20,
            "implicit_sample": True
        },
        kw_bench={"doe_num": 15, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        ref_pf="Hartmann3_MV_Uniform_Ind_0.15_PF.txt",
        kw_metrics={"Log_Hv": {"ref_point": [0.12358115, 0.21843887]}},
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
