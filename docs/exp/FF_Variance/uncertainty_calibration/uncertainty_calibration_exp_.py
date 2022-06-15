"""
Comparison of RFF and QFF on providing accurate robustness measure
"""

import argparse
import os

import gpflow
import joblib
import numpy as np
import tensorflow as tf
from gpflow.kernels import RBF
from scipy.stats import kstest, wasserstein_distance
from sklearn.metrics import mutual_info_score
from tensorflow_probability import distributions as tfd

from docs.exp.FF_Variance.mc_comparison_of_input_and_spectral_density.cfg import exp_cfgs
from trieste.acquisition.sampler import QFFVarianceSampler, RFFVarianceSampler
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.config import ModelConfig
from trieste.space import Box
from trieste.types import TensorType
from trieste.utils import DEFAULTS


def main(**kwargs):
    """
    Conduct An experiment comparing:
    - MC-based Variance on GP input
    - Spectral Density on GP
    The variant to be investigated is:
    problem (dimensionality)
    - 1d SinLinForrester
    - 2d GMM
    - 3d ...
    - 4d ...

    - num_training_samples (we keep this fixed as we don't wanna make our plot too complicated)
    - num_fourier_features
    - input_uncertainty_variates
    """
    if kwargs != {}:
        input_d = kwargs["d"]
        nx_tr = kwargs["nx_tr"]
        ff_num_list = kwargs["ff_n"]
        exp_repeat = kwargs["r"]
        noise_param_ = kwargs["n_param"]
        noise_type_ = kwargs["noise_type"]
        robustness_measure = kwargs["robustness_measure"]
        base_exhausted_mc_approax_per_input_dim = kwargs["mc_i_num"]
        max_batch = kwargs["b_xt"]
        output_samples = kwargs.get("op_s", 128)
        # file_info_prefix = kwargs["file_info_prefix"] if "file_info_prefix" in kwargs else ""
    else:  # use argparse
        parser = argparse.ArgumentParser()

        parser.add_argument("-d", type=int)
        parser.add_argument(
            "-nx_tr", nargs="+", help="<Required> Set flag", required=True, type=int
        )
        parser.add_argument("-r", "--repeat", type=int)
        parser.add_argument("-ff_n", nargs="+", help="<Required> Set flag", required=True, type=int)
        parser.add_argument(
            "-n_param", nargs="+", help="<Required> Set flag", required=True, type=float
        )
        parser.add_argument("-n_type", type=str)
        parser.add_argument("-r_msr", type=str)
        parser.add_argument("-mc_i_num", type=int)
        parser.add_argument("-b_xt", type=int)
        parser.add_argument("-op_s", type=int, default=128)

        # optional args
        parser.add_argument("-fp", "--file_info_prefix", default="", type=str)

        _args = parser.parse_args()
        tf.print(_args)
        input_d = _args.d
        nx_tr = _args.nx_tr
        exp_repeat = _args.repeat
        ff_num_list = _args.ff_n
        noise_param_ = _args.n_param
        noise_type_ = _args.n_type
        robustness_measure = _args.r_msr
        base_exhausted_mc_approax_per_input_dim = _args.mc_i_num
        max_batch = _args.b_xt
        output_samples = _args.op_s
        # file_info_prefix = _args.file_info_prefix if _args.file_info_prefix else ""
    search_space = Box([0.0] * input_d, [1.0] * input_d)

    def extract_robustness_distribution_from_gp_full_posterior(
        input, model, noisy_dist, mc_input_num, robustness_msr
    ):
        """
        Approximate robustness by exhausted MC sampling on GP posterior
        :param input
        :param model
        :param mc_input_num
        :param noisy_dist
        """
        input_noisy_dists = noisy_dist.sample(mc_input_num)
        input_dists = tf.expand_dims(input, -2) + tf.cast(
            input_noisy_dists, dtype=input.dtype
        )  # [N, mc, dim]
        # [N, mc, dim], [N, 1, mc, mc]
        jitter: float = DEFAULTS.JITTER
        jitter_identity = jitter * tf.eye(mc_input_num, dtype=input.dtype)
        Xstar_dists = []
        for input in input_dists:  # [mc, dim]
            means, covs = model.predict_joint(input)  # [mc, 1], [mc, mc]
            trajectory_sampler = tfd.MultivariateNormalFullCovariance(
                tf.squeeze(means), tf.squeeze(covs, 0) + jitter_identity
            )
            sampled_dists = trajectory_sampler.sample(output_samples)  # [post_sample_num, mc]?
            if robustness_msr == "mean":
                Xstar_dists.append(tf.math.reduce_mean(sampled_dists, -1))  # [post_sample_num, mc]
            else:
                assert robustness_msr == "variance"
                Xstar_dists.append(
                    tf.math.reduce_variance(sampled_dists, -1)
                )  # [post_sample_num, mc]

        return tf.stack(Xstar_dists)  # [N, post_sample_num]

    def compute_rff_distribution(
        input, model, training_dataset, samples_num, num_ff_features, noise_type, noise_info
    ):
        """
        :param input
        :param model
        :param samples_num
        :param num_ff_features
        """
        robustness_sampler = RFFVarianceSampler(
            noise_type, model, training_dataset, num_features=num_ff_features, noise=noise_info
        )

        robustness_samples = []
        for i in range(samples_num):
            if robustness_measure == "mean":
                robustness_samples.append(
                    robustness_sampler.get_mean_trajectory(
                        max_batch_element_num=max_batch, sample_method="standard"
                    )(input)
                )
            else:
                assert robustness_measure == "variance"
                robustness_samples.append(
                    robustness_sampler.get_var_trajectory(
                        max_batch_element_num=max_batch, sample_method="standard"
                    )(input)
                )
        return tf.transpose(
            tf.squeeze(tf.convert_to_tensor(robustness_samples), -1)
        )  # [N, post_sample_num]

    def compute_qff_distribution(
        input, model, training_dataset, samples_num, num_ff_features, noise_type, noise_info
    ):
        """
        :param input
        :param model
        :param samples_num
        :param num_ff_features
        """
        var_sampler = QFFVarianceSampler(
            noise_type, model, training_dataset, num_features=num_ff_features, noise=noise_info
        )
        robustness_samples = []
        for i in range(samples_num):
            if robustness_measure == "mean":
                robustness_samples.append(
                    var_sampler.get_mean_trajectory(
                        max_batch_element_num=max_batch, sample_method="standard"
                    )(input)
                )
            else:
                assert robustness_measure == "variance"
                robustness_samples.append(
                    var_sampler.get_var_trajectory(
                        max_batch_element_num=max_batch, sample_method="standard"
                    )(input)
                )
        return tf.transpose(
            tf.squeeze(tf.convert_to_tensor(robustness_samples), -1)
        )  # [N, post_sample_num]

    def conduct_experiment(
        expand_search_space: Box,
        num_train_samples: int,
        num_features: int,
        noisy_distribution: tfd,
        noise_type: str,
        noise_info: TensorType,
    ):
        """
        Compute the log10 Wassertein distance between the weight space approximated GP and the exact GP,
        and between the hybrid-rule approximated GP and the exact GP.

        :param expand_search_space: the search space expanded with .95 confidence interval of noise distribution
        :param num_train_samples: The number of training samples.
        :param num_features: The number of feature functions.
        :param noisy_distribution
        :param noise_type: str, can only be 'normal' or 'uniform'
        :param noise_info: hyperparam of noise distribution, used for FF_sampler

        :return: The log10 Wasserstein distances for both approximations.
        """
        # exact kernel

        noise_variance = 1e-5

        # exact kernel
        exact_kernel = kernel_class(lengthscales=[0.3] * input_d)

        X = expand_search_space.sample(num_train_samples)

        # Sample observations from prior multivariate Normal
        kXX = exact_kernel.K(X)
        kXX_plus_noise_var = tf.linalg.set_diag(kXX, tf.linalg.diag_part(kXX) + noise_variance)
        lXX = tf.linalg.cholesky(kXX_plus_noise_var)
        y = tf.matmul(lXX, tf.random.normal([num_train_samples, 1], dtype=X.dtype))

        gpr = gpflow.models.GPR((X, y), exact_kernel, noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        m = create_model(
            ModelConfig(
                **{
                    "model": gpr,
                    "optimizer": gpflow.optimizers.Scipy(),
                    "optimizer_args": {"minimize_args": {"options": dict(maxiter=200)}},
                }
            )
        )
        m.optimize(Dataset(X, y))

        # X_star = exp_cfg["search_space"].sample_sobol(num_test_samples)
        X_star = search_space.sample_sobol(num_test_samples)

        # identify mean and covariance of the analytic GPR posterior when using the weight space approximated kernel
        rff_robustness_Xstar_dist = compute_rff_distribution(
            X_star,
            model=m,
            training_dataset=Dataset(X, y),
            samples_num=output_samples,
            num_ff_features=num_features,
            noise_type=noise_type,
            noise_info=noise_info,
        )

        # identify mean and covariance using the hybrid approximation
        qff_robustness_Xstar_dist = compute_qff_distribution(
            X_star,
            model=m,
            training_dataset=Dataset(X, y),
            samples_num=output_samples,
            num_ff_features=int(np.round(num_features ** (1 / input_d))),
            noise_type=noise_type,
            noise_info=noise_info,
        )

        mc_robustness_Xstar_ground_truth = extract_robustness_distribution_from_gp_full_posterior(
            X_star,
            model=m,
            noisy_dist=noisy_distribution,
            mc_input_num=min(base_exhausted_mc_approax_per_input_dim * 2 ** input_d, 6000),
            robustness_msr=robustness_measure,
        )

        return (
            rff_robustness_Xstar_dist,
            qff_robustness_Xstar_dist,
            mc_robustness_Xstar_ground_truth,
        )
        # TODO: Maybe we should stor all raw samples, and later use some metric as our wish
        # return the log Wasserstein distances for both approximations
        # # compute log10 Wasserstein distance between the exact solution and the weight space approximation

        # use alpha=0.01
        # ws_dist_rff = []
        # for rff_var_dist, mc_var_dist in zip(rff_robustness_Xstar_dist, mc_robustness_Xstar_ground_truth):
        #     ks_statistic = kstest(mc_var_dist, rff_var_dist)[0]
        #     critical_threshold = 1.628 * np.sqrt(2 * output_samples/output_samples ** 2) # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#:~:text=For%20large%20samples,common%20levels%20of
        #     ws_dist_rff.append(ks_statistic <= critical_threshold)

        # # # compute log10 Wassertein distance between the exact solution and the hybrid approximation
        # ws_dist_qff = []
        # for qff_var_dist, mc_var_dist in zip(qff_robustness_Xstar_dist, mc_robustness_Xstar_ground_truth):
        #     ks_statistic = kstest(mc_var_dist, qff_var_dist)[0]
        #     critical_threshold = 1.628 * np.sqrt(2 * output_samples/output_samples ** 2)
        #     ws_dist_qff.append(ks_statistic <= critical_threshold)

        # return the log Wasserstein distances for both approximations
        # return np.sum(np.asarray(ws_dist_rff))/len(ws_dist_rff), np.sum(np.asarray(ws_dist_qff))/len(ws_dist_qff)

    def conduct_experiment_for_multiple_runs(
        expand_search_space,
        num_train_samples,
        num_features,
        noisy_distribution,
        noise_dist_type,
        noise_dist_info,
    ) -> tuple:
        """
        Conduct the experiment as specified above `num_experiment_runs` times and identify the quartiles for
        the log10 Wassertein distance between the weight space approximated GP and the exact GP,
        and between the hybrid-rule approximated GP and the exact GP.

        :param expand_search_space
        :param num_train_samples: The number of training samples.
        :param num_features: The number of feature functions.
        :param noisy_distribution
        :param noise_dist_info
        :param noise_dist_type

        :return: RMSE compared with ground truth
        """
        list_of_rff_samples = []
        list_of_qff_samples = []
        list_of_mc_samples = []
        for _ in range(num_experiment_runs):
            print(f"Exp: {_} times")
            rff_samples, qff_samples, mc_samples = conduct_experiment(
                expand_search_space=expand_search_space,
                num_train_samples=num_train_samples,
                num_features=num_features,
                noisy_distribution=noisy_distribution,
                noise_type=noise_dist_type,
                noise_info=noise_dist_info,
            )
            list_of_rff_samples.append(rff_samples)
            list_of_qff_samples.append(qff_samples)
            list_of_mc_samples.append(mc_samples)
        return list_of_rff_samples, list_of_qff_samples, list_of_mc_samples

    def conduct_experiment_for_different_num_features(
        expand_search_space, noisy_dist, nts, noisy_type, noisy_info, num_features
    ):
        """
        Conduct the experiment as specified above for a different number of feature functions, and store
        the results in lists of lists.

        :param expand_search_space
        :param noisy_dist
        :param nts: number of training samples
        :param noisy_type
        :param noisy_info
        :param num_features

        :return: Lists of lists of quartiles of the log10 Wasserstein distance for both approximations.
        """
        rff_results, qff_results, mc_results = conduct_experiment_for_multiple_runs(
            expand_search_space=expand_search_space,
            num_train_samples=nts,
            num_features=num_features,
            noisy_distribution=noisy_dist,
            noise_dist_type=noisy_type,
            noise_dist_info=noisy_info,
        )

        return rff_results, qff_results, mc_results

    for ff_num in ff_num_list:
        print(f"ff_num: {ff_num}")
        # settings that are fixed across experiments
        # Exp hyper-param
        kernel_class = RBF  # choose alternatively kernel_class = Matern52
        num_test_samples = (
            10 * 2 ** input_d
        )  # number of test samples for evaluation (1024 in the paper)
        num_experiment_runs = exp_repeat  # number of experiment repetitions (64 in the paper)
        for num_training_data in nx_tr:
            # We use max(wasserstein_distance(X_test)) as a way of measuring the max error
            for noise_param__ in noise_param_:
                print(
                    f"Conduct Exp on FF: {ff_num}, n_tr: {num_training_data}, n_param: {noise_param__}"
                )
                if noise_type_ == "normal":
                    # !!!!!!!! important note: noise_param__ is normal input dist is VARIANCE
                    input_uncertainty_dist = tfd.Normal(
                        tf.zeros(input_d, dtype=tf.float64),
                        tf.cast(tf.sqrt(noise_param__), dtype=tf.float64)
                        * tf.ones(input_d, dtype=tf.float64),
                    )
                    noise_info_ = noise_param__ * tf.eye(input_d, dtype=tf.float64)
                    delta = tf.cast(1.96 * tf.sqrt(noise_param__), dtype=tf.float64)
                    expanded_search_space = Box(
                        search_space.lower - delta, search_space.upper + delta
                    )
                else:
                    assert noise_type_ == "uniform"
                    input_uncertainty_dist = tfd.Uniform(
                        tf.zeros(shape=input_d, dtype=tf.float64)
                        - noise_param__ * tf.ones(shape=input_d, dtype=tf.float64),
                        tf.zeros(shape=input_d, dtype=tf.float64)
                        + noise_param__ * tf.ones(shape=input_d, dtype=tf.float64),
                    )
                    noise_info_ = noise_param__ * tf.ones(shape=input_d, dtype=tf.float64)
                    delta = 0.95 * noise_param__
                    expanded_search_space = Box(
                        search_space.lower - delta, search_space.upper + delta
                    )

                res_rff, res_qff, res_mc = conduct_experiment_for_different_num_features(
                    expand_search_space=expanded_search_space,
                    noisy_dist=input_uncertainty_dist,
                    nts=num_training_data,
                    noisy_type=noise_type_,
                    noisy_info=noise_info_,
                    num_features=ff_num,
                )

                joblib.dump(
                    res_rff,
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "exp_res",
                        f"input_dim_{input_d}",
                        robustness_measure,
                        f"{robustness_measure}_RFF_FF_{ff_num}_{noise_type_}_noise_{noise_param__}_ntr_{num_training_data}.jblb",
                    ),
                    3,
                )
                joblib.dump(
                    res_qff,
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "exp_res",
                        f"input_dim_{input_d}",
                        robustness_measure,
                        f"{robustness_measure}_QFF_FF_{ff_num}_{noise_type_}_noise_{noise_param__}_ntr_{num_training_data}.jblb",
                    ),
                    3,
                )
                joblib.dump(
                    res_mc,
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "exp_res",
                        f"input_dim_{input_d}",
                        robustness_measure,
                        f"{robustness_measure}_MC_FF_{ff_num}_{noise_type_}_noise_{noise_param__}_ntr_{num_training_data}.jblb",
                    ),
                    3,
                )


if __name__ == "__main__":
    main()
    # main(
    #     d=2, cfg="GMM", nx_tr=[20], ff_n=[100, 400, 900, 1600, 2500], r=64, noise_type="uniform",
    #     n_param=[0.05, 0.1, 0.2]
    # )
    # main(
    #     d=1, cfg="SinLinear", nx_tr=[10], ff_n=[50], r=8, noise_type="uniform",
    #     n_param=[0.05, 0.1, 0.2], robustness_measure = 'variance', mc_i_num=500, b_xt=10,
    #     op_s=256
    # )
    # main(
    #     d=3, cfg="Hartmann3", nx_tr=[30], ff_n=[2000, 4000, 5000], r=8, noise_type="uniform",
    #     n_param=[0.05, 0.1, 0.2], robustness_measure='mean', mc_i_num=500, b_xt=10
    # )
    # main(d=4, cfg='Shekel4', nx_tr = [40], ff_n = [625, 1296, 2401, 4096, 6561], r=64, noise_type="uniform",
    #     n_param=[0.05, 0.1, 0.2], robustness_measure='variance', mc_i_num=2000, b_xt=10
    # )
