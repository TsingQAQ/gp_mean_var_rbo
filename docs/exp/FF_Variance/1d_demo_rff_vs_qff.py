"""
This is a demo of Comparison of RFF vs QFF based mean and variance
"""

import gpflow
import matplotlib
import matplotlib.pyplot as plt

# %%
import numpy as np
import tensorflow as tf

tf.random.set_seed(1821)

matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}" r"\usepackage{amsfonts}"]
from trieste.objectives.single_objectives import SingleObjectiveTestProblem

# %%
from trieste.space import Box


class func(SingleObjectiveTestProblem):
    bounds = [[0], [1]]
    dim = 1

    def objective(self):
        return h


def h(x):
    return -20 * (
        0.5 - 0.8 * tf.exp(-(((x - 0.35) / 0.25) ** 2)) - tf.exp(-(((x - 0.8) / 0.05) ** 2))
    )


ds = Box([0.1], [0.9])
X = ds.sample(18)
# X = tf.concat([X, tf.constant([[0.74], [0.75]], dtype=X.dtype)], 0)
obj = func().objective()
Y = obj(X)

## generate test points for prediction
xx = np.linspace(0, 1, 200).reshape(200, 1)  # test points must be of shape (N, D)


import tensorflow_probability as tfp

## Add predict g by Kernel MC sampling (NOT INCLUDING THE DIAGONAL PART)
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.config import ModelConfig

single_obj_data = Dataset(X, Y)
variance = tf.math.reduce_variance(single_obj_data.observations)
kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * 1)

# jitter = gpflow.kernels.White(1e-12)
gpr = gpflow.models.GPR(
    (single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1e-5
)
prior_scale = tf.cast(1.09, dtype=tf.float64)
kernel.lengthscales.prior = tfp.distributions.LogNormal(
    tf.math.log(kernel.lengthscales), prior_scale
)

gpflow.utilities.set_trainable(gpr.likelihood, False)

m2 = create_model(
    ModelConfig(
        **{
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {"minimize_args": {"options": dict(maxiter=100)}},
        }
    )
)

m2.optimize(single_obj_data)

from trieste.acquisition.sampler import QFFVarianceSampler, RFFVarianceSampler
from trieste.data import Dataset

noise = 0.05
rff_sampler = RFFVarianceSampler(
    "uniform", m2, Dataset(X, Y), num_features=256, noise=tf.constant([noise])
)
qff_sampler = QFFVarianceSampler(
    "uniform", m2, Dataset(X, Y), num_features=256, noise=tf.constant([noise])
)

fig, axs = plt.subplots(2, 3, figsize=(12, 4.5))

fmean, fvar = m2.predict(xx)
# plt.figure(figsize=(10, 5))
axs[0, 0].plot(X, Y, "kx", mew=2)
axs[0, 0].plot(xx, fmean, "C0", lw=2, linestyle="--")
axs[0, 0].fill_between(
    xx[:, 0],
    fmean[:, 0] - 1.96 * np.sqrt(fvar[:, 0]),
    fmean[:, 0] + 1.96 * np.sqrt(fvar[:, 0]),
    color="C0",
    alpha=0.2,
    label="$f$ GP Posterior",
    zorder=3
)
f_sample = rff_sampler.get_trajectory(10)(xx)
for i in range(10):
    axs[0, 0].plot(xx, f_sample[i], color="C0", linewidth=0.5, alpha=0.5, zorder=5)
# f_sample = rff_sampler.get_trajectory(get_mean=True)
# axs[0, 0].plot(xx, f_sample(xx)[0], color="C0", linewidth=2, alpha=1)
# axs.flat[0].set(
#     xlabel=r"$\boldsymbol{x} \in \mathbb{R}^D$", ylabel=r"RFF based $f(\boldsymbol{x})$ sample"
# )
axs[0, 0].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
axs[0, 0].set_ylabel(r"RFF based $f(\boldsymbol{x})$ samples", fontsize=12)
axs[0, 0].grid(True, color="w", linestyle="-", linewidth=2, zorder=0)
axs[0, 0].patch.set_facecolor("0.85")

axs[0, 0].spines["top"].set_visible(False)
axs[0, 0].spines["right"].set_visible(False)
axs[0, 0].spines["left"].set_visible(False)
axs[0, 0].spines["bottom"].set_visible(False)
# plot mean
# f_sample = rff_sampler.get_trajectory(get_mean=True)
# axs[0, 0].plot(xx, f_sample(xx)[0], color="C0", linewidth=1, alpha=1)
axs[0, 0].tick_params(labelsize=15)
axs[0, 1].grid(True, color="w", linestyle="-", linewidth=2)
axs[0, 1].patch.set_facecolor("0.85")

axs[0, 1].spines["top"].set_visible(False)
axs[0, 1].spines["right"].set_visible(False)
axs[0, 1].spines["left"].set_visible(False)
axs[0, 1].spines["bottom"].set_visible(False)


axs[0, 1].plot(X, Y, "kx", mew=2)
axs[0, 1].plot(xx, fmean, "C0", lw=2, linestyle="--")
axs[0, 1].fill_between(
    xx[:, 0],
    fmean[:, 0] - 1.96 * np.sqrt(fvar[:, 0]),
    fmean[:, 0] + 1.96 * np.sqrt(fvar[:, 0]),
    color="C0",
    alpha=0.2,
    label="$f$ GP Posterior",
    zorder=3
)
f_sample = rff_sampler.get_mean_trajectory(15)(xx)
for i in range(15):
    axs[0, 1].plot(xx, f_sample[i], color="C0", linewidth=0.5, alpha=0.5, zorder=5)
# f_sample = rff_sampler.get_mean_trajectory(get_mean=True)
# axs[0, 1].plot(xx, f_sample(xx)[0], color="C0", linewidth=2, alpha=1)
# axs.flat[1].set(
#     xlabel=r"$\boldsymbol{x} \in \mathbb{R}^D$", ylabel=r"RFF based $J(\boldsymbol{x})$ sample"
# )
axs[0, 1].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
axs[0, 1].set_ylabel(r"RFF based $\mathbb{J}(\boldsymbol{x})$ samples", fontsize=12)
axs[0, 1].tick_params(labelsize=15)
axs[0, 1].grid(True, color="w", linestyle="-", linewidth=2, zorder=0)
axs[0, 1].patch.set_facecolor("0.85")

axs[0, 1].spines["top"].set_visible(False)
axs[0, 1].spines["right"].set_visible(False)
axs[0, 1].spines["left"].set_visible(False)
axs[0, 1].spines["bottom"].set_visible(False)

axs[0, 2].plot(X, Y, "kx", mew=2)
axs[0, 2].plot(xx, fmean, "C0", lw=2, linestyle="--")
axs[0, 2].fill_between(
    xx[:, 0],
    fmean[:, 0] - 1.96 * np.sqrt(fvar[:, 0]),
    fmean[:, 0] + 1.96 * np.sqrt(fvar[:, 0]),
    color="C0",
    alpha=0.2,
    label="$f$ GP Posterior",
    zorder=3
)
v_sample = rff_sampler.get_var_trajectory(15)(xx)
for i in range(15):
    axs[0, 2].plot(xx, v_sample[i], color="C0", linewidth=0.5, alpha=0.5, zorder=5)
# v_sample = rff_sampler.get_var_trajectory(get_mean=True)
# axs[0, 2].plot(xx, v_sample(xx)[0], color="C0", linewidth=2, alpha=1)
# axs.flat[2].set(
#     xlabel=r"$\boldsymbol{x} \in \mathbb{R}^D$",
#     ylabel=r"RFF based $\mathbb{V}(\boldsymbol{x})$ sample",
# )
axs[0, 2].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
axs[0, 2].set_ylabel(r"RFF based $\mathbb{V}(\boldsymbol{x})$ samples", fontsize=12)
axs[0, 2].tick_params(labelsize=15)
axs[0, 2].grid(True, color="w", linestyle="-", linewidth=2, zorder=0)
axs[0, 2].patch.set_facecolor("0.85")

axs[0, 2].spines["top"].set_visible(False)
axs[0, 2].spines["right"].set_visible(False)
axs[0, 2].spines["left"].set_visible(False)
axs[0, 2].spines["bottom"].set_visible(False)
# =================
axs[1, 0].plot(X, Y, "kx", mew=2)
axs[1, 0].plot(xx, fmean, "C1", lw=2, linestyle="--")
axs[1, 0].fill_between(
    xx[:, 0],
    fmean[:, 0] - 1.96 * np.sqrt(fvar[:, 0]),
    fmean[:, 0] + 1.96 * np.sqrt(fvar[:, 0]),
    color="C1",
    alpha=0.2,
    label="$f$ GP Posterior",
)
f_sample = qff_sampler.get_trajectory(15)(xx)
for i in range(15):
    axs[1, 0].plot(xx, f_sample[i], color="C1", linewidth=0.5, alpha=0.5)
# f_sample = qff_sampler.get_trajectory(get_mean=True)
# axs[1, 0].plot(xx, f_sample(xx)[0], color="C1", linewidth=2, alpha=1)
# axs.flat[3].set(
#     xlabel=r"$\boldsymbol{x} \in \mathbb{R}^D$", ylabel=r"QFF based $f(\boldsymbol{x})$ sample"
# )
axs[1, 0].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
axs[1, 0].set_ylabel(r"QFF based $f(\boldsymbol{x})$ samples", fontsize=12)
axs[1, 0].tick_params(labelsize=15)
axs[1, 0].grid(True, color="w", linestyle="-", linewidth=2)
axs[1, 0].patch.set_facecolor("0.85")

axs[1, 0].spines["top"].set_visible(False)
axs[1, 0].spines["right"].set_visible(False)
axs[1, 0].spines["left"].set_visible(False)
axs[1, 0].spines["bottom"].set_visible(False)


axs[1, 1].plot(X, Y, "kx", mew=2)
axs[1, 1].plot(xx, fmean, "C1", lw=2, linestyle="--")
axs[1, 1].fill_between(
    xx[:, 0],
    fmean[:, 0] - 1.96 * np.sqrt(fvar[:, 0]),
    fmean[:, 0] + 1.96 * np.sqrt(fvar[:, 0]),
    color="C1",
    alpha=0.2,
    label="$f$ GP Posterior",
    zorder=3
)
f_sample = qff_sampler.get_mean_trajectory(15)(xx)
for i in range(15):
    axs[1, 1].plot(xx, f_sample[i], color="C1", linewidth=0.5, alpha=0.5, zorder=5)
# f_sample = qff_sampler.get_mean_trajectory(get_mean=True)
# axs[1, 1].plot(xx, f_sample(xx)[0], color="C1", linewidth=2, alpha=1)
# axs.flat[4].set(
#     xlabel=r"$\boldsymbol{x} \in \mathbb{R}^D$", ylabel=r"QFF based $J(\boldsymbol{x})$ sample"
# )
axs[1, 1].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
axs[1, 1].set_ylabel(r"QFF based $\mathbb{J}(\boldsymbol{x})$ samples", fontsize=12)
axs[1, 1].tick_params(labelsize=15)
axs[1, 1].grid(True, color="w", linestyle="-", linewidth=2, zorder=0)
axs[1, 1].patch.set_facecolor("0.85")

axs[1, 1].spines["top"].set_visible(False)
axs[1, 1].spines["right"].set_visible(False)
axs[1, 1].spines["left"].set_visible(False)
axs[1, 1].spines["bottom"].set_visible(False)

axs[1, 2].plot(X, Y, "kx", mew=2)
axs[1, 2].plot(xx, fmean, "C1", lw=2, linestyle="--")
axs[1, 2].fill_between(
    xx[:, 0],
    fmean[:, 0] - 1.96 * np.sqrt(fvar[:, 0]),
    fmean[:, 0] + 1.96 * np.sqrt(fvar[:, 0]),
    color="C1",
    alpha=0.2,
    label="$f$ GP Posterior",
    zorder=3
)

v_sample = qff_sampler.get_var_trajectory(15)(xx)
for i in range(15):
    axs[1, 2].plot(xx, v_sample[i], color="C1", linewidth=0.5, alpha=0.5, zorder=5)
# v_sample = qff_sampler.get_var_trajectory(get_mean=True)
# axs[1, 2].plot(xx, v_sample(xx)[0], color="C1", linewidth=2, alpha=1)
# axs.flat[5].set(
#     xlabel=r"$\boldsymbol{x} \in \mathbb{R}^D$",
#     ylabel=r"QFF based $\mathbb{V}(\boldsymbol{x})$ sample",
# )
axs[1, 2].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
axs[1, 2].set_ylabel(r"QFF based $\mathbb{V}(\boldsymbol{x})$ samples", fontsize=12)
axs[1, 2].tick_params(labelsize=15)
axs[1, 2].grid(True, color="w", linestyle="-", linewidth=2, zorder=0)
axs[1, 2].patch.set_facecolor("0.85")

axs[1, 2].spines["top"].set_visible(False)
axs[1, 2].spines["right"].set_visible(False)
axs[1, 2].spines["left"].set_visible(False)
axs[1, 2].spines["bottom"].set_visible(False)

plt.tight_layout()

plt.savefig("QFF_vs_RFF_new.png", dpi=500)
# plt.show()
