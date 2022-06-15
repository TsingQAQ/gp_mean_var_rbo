import matplotlib
from matplotlib import pyplot as plt

matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}" r"\usepackage{amsfonts}"]
import os
from copy import deepcopy

import numpy as np


def ci(y):
    return 1.96 * y.std(axis=0) / np.sqrt(y.shape[0])


def get_performance_result_from_file(path_prefix, exp_repeat: int, file_prefix, file_suffix):
    regret_list = []
    for i in range(exp_repeat):
        res_path = os.path.join(path_prefix, "".join([file_prefix, f"_{i}_", file_suffix, ".txt"]))
        try:
            regret_list.append(np.loadtxt(res_path)[None, ...])
        except:
            print(f"Cannot load {res_path}, skip")
    # print(path_prefix)
    regrets = np.concatenate(regret_list, axis=0)
    return regrets


def plot_convergence_curve_for_cfg(cfg: dict):
    path_prefix = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        cfg["which_formulation"],
        "exp_res",
        cfg["pb_name"],
    )
    legend_handles = []
    legend_label = []

    color_idx = 0
    color_list = ["C0", "C1", "C2", "C3", "C4", "C5"]
    plt.figure(figsize=cfg["plot_cfg"]["fig_size"])
    for acq_exp_label, _ in cfg["acq"].items():
        x_space = 1 if 'x_space' not in cfg else cfg['x_space']
        if acq_exp_label.split("-")[0] == "FF":
            for aux_prefix, aux_label in zip(["rff", "qff"], ["R", "Q"]):
                # for aux_prefix, aux_label in zip(['qff'], ['Q']):
                _path_prefix = os.path.join(
                    path_prefix, acq_exp_label, "".join([cfg["res_path_prefix"], "_", aux_prefix])
                )
                exp_regrets = get_performance_result_from_file(
                    _path_prefix,
                    exp_repeat=cfg["exp_repeat"],
                    file_prefix=cfg["pb_name"],
                    file_suffix=cfg["file_suffix"],
                )
                if "max_iter" in cfg.keys():
                    exp_regrets = exp_regrets[:, : cfg["max_iter"] + 1]
                if "fix_threshold" in cfg.keys():
                    exp_regrets += cfg["fix_threshold"]
                a1 = plt.plot(np.arange(exp_regrets.shape[-1]), np.percentile(exp_regrets, 50, axis=0), color=color_list[color_idx], zorder=5)
                plt.fill_between(
                    np.arange(exp_regrets.shape[-1]),
                    np.percentile(exp_regrets, 25, axis=0),
                    np.percentile(exp_regrets, 75, axis=0),
                    color=color_list[color_idx],
                    alpha=0.2,
                    label=''.join([aux_label, acq_exp_label]),
                    zorder=3
                )
                # plt.errorbar(
                #     np.arange(exp_regrets.shape[-1])[::x_space],
                #     exp_regrets.mean(axis=0)[::x_space],
                #     yerr=ci(exp_regrets)[::x_space],
                #     label="".join([aux_label, acq_exp_label]),
                #     linewidth=1.5,
                # )
                a2 = plt.fill(np.NaN, np.NaN, alpha=0.2, color=color_list[color_idx])
                legend_handles.append((a1[0], a2[0]))
                legend_label.append(''.join([aux_label, acq_exp_label]))
                plt.xlabel(
                    "Number of Iterations", fontsize=cfg["plot_cfg"].get("label_fontsize", 10)
                )
                plt.ylabel(
                    cfg["plot_cfg"]["plot_ylabel"],
                    fontsize=cfg["plot_cfg"].get("label_fontsize", 10),
                )
                color_idx += 1
        else:
            _path_prefix = os.path.join(path_prefix, acq_exp_label, cfg["res_path_prefix"])
            exp_regrets = get_performance_result_from_file(
                _path_prefix,
                exp_repeat=cfg["exp_repeat"],
                file_prefix=cfg["pb_name"],
                file_suffix=cfg["file_suffix"],
            )
            if "max_iter" in cfg.keys():
                exp_regrets = exp_regrets[:, : cfg["max_iter"] + 1]
            if "fix_threshold" in cfg.keys():
                exp_regrets += cfg["fix_threshold"]
            # plt.errorbar(
            #     np.arange(exp_regrets.shape[-1])[::x_space],
            #     exp_regrets.mean(axis=0)[::x_space],
            #     yerr=ci(exp_regrets)[::x_space],
            #     label=acq_exp_label,
            #     linewidth=1.5,
            # )
            a1 = plt.plot(np.arange(exp_regrets.shape[-1]), np.percentile(exp_regrets, 50, axis=0), color=color_list[color_idx], zorder=5)
            plt.fill_between(
                np.arange(exp_regrets.shape[-1]),
                np.percentile(exp_regrets, 25, axis=0),
                np.percentile(exp_regrets, 75, axis=0),
                color=color_list[color_idx],
                alpha=0.2,
                label=acq_exp_label,
                zorder=3
            )
            a2 = plt.fill(np.NaN, np.NaN, alpha=0.2, color=color_list[color_idx])
            legend_handles.append((a1[0], a2[0]))
            legend_label.append(acq_exp_label)
            # plt.xlabel("Numer of Iterations", fontsize=cfg["plot_cfg"].get("label_fontsize", 10))
            # plt.ylabel(
            #     cfg["plot_cfg"]["plot_ylabel"], fontsize=cfg["plot_cfg"].get("label_fontsize", 10)
            # )
            color_idx += 1
    if cfg["plot_cfg"]["log_y"] == True:
        plt.yscale("log")
    # plt.yticks([1e-1, 1e-2, 1e-3])
    plt.xticks(fontsize=cfg["plot_cfg"].get("tick_fontsize", 5))
    plt.yticks(fontsize=cfg["plot_cfg"].get("tick_fontsize", 5))
    plt.legend(fontsize=cfg["plot_cfg"].get("lgd_fontsize", 10))
    # plt.legend(
    #     legend_handles,
    #     legend_label,
    #     fontsize=cfg['plot_cfg'].get('lgd_fontsize', 10),
    # )
    plt.title(cfg["plot_cfg"]["title"], fontsize=cfg["plot_cfg"].get("title_fontsize", 10))
    plt.grid(zorder=1)
    plt.tight_layout()
    # plt.show(block=True)
    plt.grid(True, color="w", linestyle="-", linewidth=2)
    plt.gca().patch.set_facecolor("0.85")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.savefig("_".join([cfg["pb_name"], cfg["which_formulation"], ''.join([cfg["which_noise"], ".png"])]), dpi=300)

path_res = {"normal": os.path.join("normal", "q1"), "uniform": os.path.join("uniform", "q1")}
sinlinear_mo_titles = {
    "normal": r"SinLinear, $\xi \sim \mathcal{N}(0, 0.001)$",
    "uniform": r"SinLinear, $\xi \sim U(-0.05, 0.05)$",
}
sinlinear_mo_config = {
    "pb_name": "SinLinear",
    "which_formulation": "mo_mean_var_exp",
    "acq": {"FF-MV-qEHVI": {}, "Random": {}, "US": {}, "MO-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Log_Hv_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 20,
        "plot_ylabel": "Log Hv Difference",
        "title": sinlinear_mo_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
}

forrester_mo_titles = {
    "normal": r"Forrester, $\xi \sim \mathcal{N}(0, 0.005)$",
    "uniform": r"Forrester, $\xi \sim U(-0.1, 0.1)$",
}
forrester_mo_config = {
    "pb_name": "Forrester",
    "which_formulation": "mo_mean_var_exp",
    "acq": {"FF-MV-qEHVI": {}, "Random": {}, "US": {}, "MO-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Log_Hv_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 20,
        "plot_ylabel": "Log Hv Difference",
        "title": forrester_mo_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
}
branin_mo_titles = {
    "normal": r"Branin, $\xi \sim \mathcal{N}(0, 0.01\boldsymbol{I}_2)$",
    "uniform": r"Branin, $\xi \sim U(-[0.1, 0.01], [0.1, 0.01])$",
}
branin_mo_config = {
    "pb_name": "Branin",
    "which_formulation": "mo_mean_var_exp",
    "acq": {"FF-MV-qEHVI": {}, "Random": {}, "US": {}, "MO-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Log_Hv_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 20,
        "plot_ylabel": "Log Hv Difference",
        "title": branin_mo_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "max_iter": 50,
    "which_noise": "uniform",
}

conceptual_wing_mo_titles = {
    "normal": r"Conceptual Low Drag Wing Design",
    "uniform": r"Conceptual Low Drag Wing Design",
}
conceptual_wing_mo_config = {
    "pb_name": "Conceptual_Low_Drag_Wing_Design",
    "which_formulation": "mo_mean_var_exp",
    "acq": {"FF-MV-qEHVI": {}, "Random": {}, "US": {}, "MO-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Log_Hv_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 20,
        "plot_ylabel": "Log Hv Difference",
        "title": r"Conceptual Low Drag Wing Design",
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "max_iter": 50,
}
hartmann3_mo_titles = {
    "normal": r"Hartmann3, $\xi \sim \mathcal{N}(0, 0.01\boldsymbol{I}_3)$",
    "uniform": r"Hartmann3, $\xi \sim U(-\boldsymbol{0.15}, \boldsymbol{0.15})$",
}
hartmann3_mo_config = {
    "pb_name": "Hartmann_3",
    "which_formulation": "mo_mean_var_exp",
    "acq": {"FF-MV-qEHVI": {}, "Random": {}, "US": {}, "MO-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Log_Hv_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 20,
        "plot_ylabel": "Log Hv Difference",
        "title": hartmann3_mo_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
}
# -------------------------
sinlinear_vc_titles = {
    "normal": r"SinLinear, $\xi \sim \mathcal{N}(0, 0.001), C_v=0.14$",
    "uniform": r"SinLinear, $\xi \sim U(-0.05, 0.05), C_v=0.14$",
}
sinlinear_vc_config = {
    "pb_name": "SinLinear",
    "which_formulation": "var_as_con_acq_exp",
    "acq": {"FF-MV-qECI": {}, "Random": {}, "US": {}, "CO-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_ylabel": "Inference Regret",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 20,
        "plot_ylabel": "Utility Gap",
        "title": sinlinear_vc_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
}

forrester_vc_titles = {
    "normal": r"Forrester, $\xi \sim \mathcal{N}(0, 0.005), C_v=1$",
    "uniform": r"Forrester, $\xi \sim U(-0.1, 0.1), C_v=1$",
}
forrester_vc_config = {
    "pb_name": "Forrester",
    "which_formulation": "var_as_con_acq_exp",
    "acq": {"FF-MV-qECI": {}, "Random": {}, "US": {}, "CO-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 20,
        "plot_ylabel": "Utility Gap",
        "title": forrester_vc_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
}  # , 'fix_threshold': 0.673

branin_vc_titles = {
    "normal": r"Branin, $\xi \sim \mathcal{N}(0, 0.01\boldsymbol{I}_2), C_v=160$",
    "uniform": r"Branin, $\xi \sim U(-\boldsymbol{0.05}, \boldsymbol{0.05}), C_v=160$",
}
branin_vc_config = {
    "pb_name": "Branin",
    "which_formulation": "var_as_con_acq_exp",
    "acq": {"FF-MV-qECI": {}, "Random": {}, "US": {}, "CO-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 20,
        "plot_ylabel": "Utility Gap",
        "title": branin_vc_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
}
hartmann3_vc_titles = {
    "normal": r"Hartmann3, $\xi \sim \mathcal{N}(0, 0.01\boldsymbol{I}_3), C_v=0.15$",
    "uniform": r"Hartmann3, $\xi \sim U(-\boldsymbol{0.1}, \boldsymbol{0.1}), C_v=0.12$",
}
hartmann3_vc_config = {
    "pb_name": "Hartmann_3",
    "which_formulation": "var_as_con_acq_exp",
    "acq": {"FF-MV-qECI": {}, "Random": {}, "US": {}, "CO-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 20,
        "plot_ylabel": "Utility Gap",
        "title": hartmann3_vc_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
}  # , 'fix_threshold': 0.05


robot_pushing_vc_titles = {
    "normal": r"Robot Pushing, $C_v=0.05$",
    "uniform": r"Robot Pushing, $C_v = 0.03$",
}
robot_pushing_vc_config = {
    "pb_name": "Robot_Pushing_3D",
    "which_formulation": "var_as_con_acq_exp",
    "acq": {"FF-MV-qECI": {}, "Random": {}, "US": {}, "CO-MVA-BO": {}},
    "res_path_prefix": "q1",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 20,
        "plot_ylabel": "Utility Gap",
        "title": r"Robot Pushing",
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
}
# -------------------------
sinlinear_smv_titles = {
    "normal": r"SinLinear, $\xi \sim \mathcal{N}(0, 0.001), \alpha=2$",
    "uniform": r"SinLinear, $\xi \sim U(0, 0.1), \alpha=2$",
}
sinlinear_smv_config = {
    "pb_name": "SinLinear",
    "which_formulation": "scalar_mean_var_exp",
    "acq": {"FF-SMV-qEI": {}, "Random": {}, "US": {}, "MT-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 20,
        "plot_ylabel": "Inference Regret",
        "title": sinlinear_smv_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
}  # , 'fix_threshold': 5e-4
forrester_smv_titles = {
    "normal": r"Forrester, $\xi \sim \mathcal{N}(0, 0.005), \alpha=5$",
    "uniform": r"Forrester, $\xi \sim U(-0.1, 0.1), \alpha=5$",
}
forrester_smv_config = {
    "pb_name": "Forrester",
    "which_formulation": "scalar_mean_var_exp",
    "acq": {"FF-SMV-qEI": {}, "Random": {}, "US": {}, "MT-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 20,
        "plot_ylabel": "Inference Regret",
        "title": forrester_smv_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
    "fix_threshold": 0.2,
}
branin_smv_titles = {
    "normal": r"Branin, $\xi \sim \mathcal{N}\left(0, \left[\begin{smallmatrix}0.01 & -0.003\\-0.003 & 0.001\end{smallmatrix}\right]\right), \alpha=0.5$",
    "uniform": r"Branin, $\xi \sim U( \pm[0.1, 0.001]), \alpha=2$",
}
branin_smv_config = {
    "pb_name": "Branin",
    "which_formulation": "scalar_mean_var_exp",
    "acq": {"FF-SMV-qEI": {}, "Random": {}, "US": {}, "MT-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 20,
        "plot_ylabel": "Inference Regret",
        "title": branin_smv_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "max_iter": 50,
    "which_noise": "uniform",
}  #  'fix_threshold': 0.35,
hartmann3_smv_titles = {
    "normal": r"Hartmann3, $\xi \sim \mathcal{N}(0, \Sigma_{\boldsymbol{\xi}}), \alpha=12$",
    "uniform": r"Hartmann3, $\xi \sim U(-\boldsymbol{0.05}, \boldsymbol{0.05}), \alpha=12$",
}
hartmann3_smv_config = {
    "pb_name": "Hartmann_3",
    "which_formulation": "scalar_mean_var_exp",
    "acq": {"FF-SMV-qEI": {}, "Random": {}, "US": {}, "MT-MVA-BO": {}},
    "res_path_prefix": path_res["uniform"],
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 20,
        "plot_ylabel": "Inference Regret",
        "title": hartmann3_smv_titles["uniform"],
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
    "which_noise": "uniform",
    "fix_threshold": 1,
}

robot_pushing_smv_titles = {
    "normal": r"Robot Pushing, $\alpha=4$",
    "uniform": r"Robot Pushing, $\alpha=2$",
}
robot_pushing_smv_config = {
    "pb_name": "Robot_Pushing_3D",
    "which_formulation": "scalar_mean_var_exp",
    "acq": {"FF-SMV-qEI": {}, "Random": {}, "US": {}, "MT-MVA-BO": {}},
    "res_path_prefix": "q1",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_Regret_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 20,
        "plot_ylabel": "Inference Regret",
        "title": r"Robot Pushing",
        "title_fontsize": 17,
        "lgd_fontsize": 12,
        "tick_fontsize": 20,
        "fig_size": (5, 3.5),
    },
}

if __name__ == "__main__":
    plot_convergence_curve_for_cfg(cfg=branin_smv_config)
    # plot_convergence_curve_for_exp('sinlinear', 'smv', 'uniform')
