# %% [markdown]
# # Robust Optimization Considering Mean and Variance

# %% [markdown]
# In real-life applications like design optimization, practitioners might be interested in an input configuration's sensitivity with respect to input uncertainty. In Bayesian Optimization (BO), being able to make sure the recommended optimal solution robust against input noise is crucial for implementing the solution in reality. In this notebook, we demonstrate how to achieve this within Trieste.

# %% [markdown]
# ## Inference Mean and Variance Using Gaussian Process

# %% [markdown]
# In practice, the robustness of a solution can be commonly characterized by robustness measures like *mean* and *variance*:

# %% [markdown]
# \begin{equation}
# \begin{aligned}
# \mathbb{J}_{\boldsymbol{\xi}}(f) &= \mathbb{E}_{\boldsymbol{\xi}}(f(\boldsymbol{x} + \boldsymbol{\xi})) \\
# \mathbb{V}_{\boldsymbol{\xi}}(f) &= \mathbb{E}_{\boldsymbol{\xi}}(f^2(\boldsymbol{x} + \boldsymbol{\xi})) - [\mathbb{E}_{\boldsymbol{\xi}}(f(\boldsymbol{x} + \boldsymbol{\xi}))]^2 
# \end{aligned}
# \end{equation}

# %% [markdown]
# Where $\boldsymbol{\xi}$ is a pre-defined additive random variable representing the noise level of the implementation.

# %% [markdown]
# Below, let's take a simple example demonstrating how the *mean* and *variance* can be inferred under Gaussian Process

# %% [markdown]
# ### Example 1: 1 Dimensional Synthetic Function

# %% [markdown]
# First we define our toy synthetic funtion here:

# %%
from trieste.objectives.single_objectives import SingleObjectiveTestProblem
import tensorflow as tf
tf.random.set_seed(1821)

class func(SingleObjectiveTestProblem):
    bounds = [[0], [1]]
    dim = 1

    def objective(self):
        return h


def h(x):
    return -20 * (
        0.5 - 0.8 * tf.exp(-(((x - 0.35) / 0.25) ** 2)) - tf.exp(-(((x - 0.8) / 0.05) ** 2))
    )

obj = func().objective()

# %%
from trieste.space import Box
ds = Box([0.1], [0.9])
X = ds.sample(18)
Y = obj(X)

# %% [markdown]
# Now let's construct our GP model

# %%
import tensorflow_probability as tfp

from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.config import ModelConfig
import gpflow

single_obj_data = Dataset(X, Y)
variance = tf.math.reduce_variance(single_obj_data.observations)
kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * 1)

gpr = gpflow.models.GPR(
    (single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1e-5
)
prior_scale = tf.cast(1.0, dtype=tf.float64)
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

# %% [markdown]
# Now consider we would be interest to specify a robustness level by using Unirom input noise :$\boldsymbol{\xi}\sim\text{U}(-0.05, 0.05)$, we can difine a spectral robustness measure sampler as below:

# %%
from trieste.acquisition.sampler import QFFVarianceSampler, RFFVarianceSampler
from trieste.data import Dataset

noise = 0.05
rff_sampler = RFFVarianceSampler(
    "uniform", m2, Dataset(X, Y), num_features=256, noise=tf.constant([noise])
)
qff_sampler = QFFVarianceSampler(
    "uniform", m2, Dataset(X, Y), num_features=256, noise=tf.constant([noise])
)


# %% [markdown]
# Now we can make approximation inference of the robustness measure posterior by simply doing sample:

# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
import numpy as np
# This two line can be used for labels if Latex has been installed 
# matplotlib.rc("text", usetex=True)
# matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}" r"\usepackage{amsfonts}"]

fig, axs = plt.subplots(2, 3, figsize=(12, 4.5))

xx = np.linspace(0, 1, 200).reshape(200, 1)
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
# axs[0, 0].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12) 
# axs[0, 0].set_ylabel(r"RFF based $f(\boldsymbol{x})$ samples", fontsize=12)
axs[0, 0].grid(True, color="w", linestyle="-", linewidth=2, zorder=0)
axs[0, 0].patch.set_facecolor("0.85")

axs[0, 0].spines["top"].set_visible(False)
axs[0, 0].spines["right"].set_visible(False)
axs[0, 0].spines["left"].set_visible(False)
axs[0, 0].spines["bottom"].set_visible(False)
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
# axs[0, 1].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
# axs[0, 1].set_ylabel(r"RFF based $\mathbb{J}(\boldsymbol{x})$ samples", fontsize=12)
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
# axs[0, 2].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
# axs[0, 2].set_ylabel(r"RFF based $\mathbb{V}(\boldsymbol{x})$ samples", fontsize=12)
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
# axs[1, 0].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
# axs[1, 0].set_ylabel(r"QFF based $f(\boldsymbol{x})$ samples", fontsize=12)
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
# axs[1, 1].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
# axs[1, 1].set_ylabel(r"QFF based $\mathbb{J}(\boldsymbol{x})$ samples", fontsize=12)
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
# axs[1, 2].set_xlabel(r"$\boldsymbol{x} \in \mathbb{R}$", fontsize=12)
# axs[1, 2].set_ylabel(r"QFF based $\mathbb{V}(\boldsymbol{x})$ samples", fontsize=12)
axs[1, 2].tick_params(labelsize=15)
axs[1, 2].grid(True, color="w", linestyle="-", linewidth=2, zorder=0)
axs[1, 2].patch.set_facecolor("0.85")

axs[1, 2].spines["top"].set_visible(False)
axs[1, 2].spines["right"].set_visible(False)
axs[1, 2].spines["left"].set_visible(False)
axs[1, 2].spines["bottom"].set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Example 2: 2 Dimensional Synthetic Example (~2-3 min to run)

# %%
from trieste.objectives.single_objectives import GMM
from trieste.space import Box
gmm_ds = GMM.bounds
gmm_obj = lambda x: -GMM().objective()(x)


gmm_X = Box(*gmm_ds).sample(50)
gmm_Y = gmm_obj(gmm_X)

# %% [markdown]
# We can first take a look what the GMM function looks like:

# %%
from util.plotting_plotly import plot_function_plotly
fig = plot_function_plotly(gmm_obj, *gmm_ds, grid_density=50)
fig.show()

# %% [markdown]
# Now we can build our GP model on this function:

# %%
import tensorflow_probability as tfp
import tensorflow as tf

from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.config import ModelConfig
import gpflow

gmm_data = Dataset(gmm_X, gmm_Y)
gmm_variance = tf.math.reduce_variance(gmm_data.observations)
gmm_kernel = gpflow.kernels.RBF(gmm_variance, lengthscales=[1.0] * 2)
gmm_gpr = gpflow.models.GPR(
    (gmm_data.query_points, gmm_data.observations), gmm_kernel, noise_variance=1e-5
)
gpflow.utilities.set_trainable(gmm_gpr.likelihood, False)

gmm_model = create_model(
    ModelConfig(
        **{
            "model": gmm_gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {"minimize_args": {"options": dict(maxiter=100)}},
        }
    )
)

gmm_model.optimize(gmm_data)

# %% [markdown]
# Now suppose we are interest in a variance distribution $\boldsymbol{\xi} \sim \mathcal{N}(\boldsymbol{0}, 1e-3I)$,
# let's take a look at what would the the variance of this function looks like by using GP to infer the its distribution

# %% [markdown]
# We firtst initialize our robustness sampler

# %%
from trieste.acquisition.sampler import QFFVarianceSampler, RFFVarianceSampler

gmm_noise = tf.constant([[1e-3, 0.00], [0.00, 1e-3]], dtype=tf.float64)
gmm_rff_sampler = RFFVarianceSampler("normal", gmm_model, gmm_data, num_features=20 ** 2, noise=gmm_noise)
gmm_qff_sampler = QFFVarianceSampler("normal", gmm_model, gmm_data, num_features=20, noise=gmm_noise)

# %%
# note: max_batch_element_num is the minibatch size used throug test data, here is used for splitting 50*50 = 2500 to small batch size
# of test data, it can be set a smaller value for more moemry efficiency.
gmm_rff_var_mean = gmm_rff_sampler.get_var_trajectory(max_batch_element_num=500, get_mean=True)

# %%
fig = plot_function_plotly(lambda at: gmm_rff_var_mean(at)[0], *gmm_ds, grid_density=50)
fig.show()

# %% [markdown]
# Note: here we use a `max_batch_element_num` to enforce efficient memory consumption, hence this may take some time. 

# %%
gmm_qff_var_mean = gmm_qff_sampler.get_var_trajectory(max_batch_element_num=500, get_mean=True)
fig = plot_function_plotly(lambda at: gmm_qff_var_mean(at)[0], *gmm_ds, grid_density=50)
fig.show()

# %% [markdown]
# ## Robust Bayesian Optimization

# %% [markdown]
# Besides manually investigating the variance, one would maybe interested directly in taking the robustness measures into account in Bayesian Optimization, here we provide 2 examples through the acquisition function we have extended in the paper.

# %% [markdown]
# ### 1-Dimensional Forrester Function (~3 min to run)

# %%
import trieste
from trieste.bayesian_optimizer import EfficientGlobalOptimization
from trieste.objectives.single_objectives import Forrester
from trieste.space import Box

# %%
forrester_func = Forrester().objective()
observer = trieste.objectives.utils.mk_observer(forrester_func)

search_space = Box(*Forrester.bounds)
num_objective = 1

# %% [markdown]
# Lets take an initial visualization of the objective function, as well as its mean and variance

# %%
from tensorflow_probability import distributions as tfd

noise_type='normal'
noise_param = tf.constant([[0.005]], dtype=tf.float64)  # Variance

noise_dist = tfd.MultivariateNormalFullCovariance(0.0, noise_param)

# %%
from matplotlib import pyplot as plt

xs = tf.linspace([0], [1], 100)
ys = forrester_func(xs)

base_sample = noise_dist.sample(10000)
def forrester_robust(input):
    input_dists = tf.expand_dims(input, -2) + tf.cast(
        base_sample, dtype=xs.dtype)  # [N, mc, 1]
    noisy_res = forrester_func(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return tf.concat([y_means, y_vars], -1)



# %% [markdown]
# We can first extract the ground truth using a NSGAII on the real function as a reference:

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo

mean_var_pf, mean_var_pf_x = moo_optimize_pymoo(forrester_robust, 1, 2, ((0.0), (1.0)), 50, 2000, return_pf_x=True)

# %%
y_means, y_vars = tf.split(forrester_robust(xs), 2, axis=-1)

plt.figure()
plt.plot(xs, ys, label='function')
plt.plot(xs, y_means, label='Mean: $\mathbb{J}$')
plt.plot(xs, y_vars, label='Variance: $\mathbb{V}$')
plt.scatter(mean_var_pf_x, forrester_func(mean_var_pf_x), label='Pareto optimal Input')
plt.legend()
plt.show()

# %%
plt.figure()
plt.scatter(mean_var_pf[:, 0], mean_var_pf[:, 1])
plt.xlabel('Mean $\mathbb{J}$')
plt.ylabel('Variance $\mathbb{V}$')
plt.show()

# %%
num_initial_points = 5
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)

import gpflow

from trieste.models.gpflow.models import GaussianProcessRegression


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance, lengthscales=[1.0])
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


model = build_model(initial_data)


# %%
from trieste.acquisition.function.robust import FF_MV_qEHVI

ff_mva_ehvi = FF_MV_qEHVI(noise_type=noise_type, noise_param = noise_param,
                          ff_method='QFF', opt_ff_num=128, mc_num=128, infer_mc_num=10000, implicit_sample=True)
rule: EfficientGlobalOptimization = EfficientGlobalOptimization(
    builder=ff_mva_ehvi, num_query_points=1, num_initial_samples=100, num_optimization_runs=1)

# %%
num_steps = 20
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)

# %% [markdown]
# Now we can visualize the results

# %% [markdown]
# In design space:

# %%
# %matplotlib notebook

# %%
import tensorflow as tf

dataset = result.try_get_final_dataset()
data_query_points = dataset.query_points
data_observations = dataset.observations

from trieste.acquisition.sampler import QFFVarianceSampler
sampler = QFFVarianceSampler(
    noise_type, model, dataset, num_features=128, noise=noise_param
)
ff_mean = sampler.get_mean_trajectory(get_mean=True)
ff_var = sampler.get_var_trajectory(get_mean=True)
plt.figure()
plt.plot(xs, ys, label='function')
plt.plot(xs, y_means, label='Mean: J')
plt.plot(xs, ff_mean(xs)[0], label='FF-Mean Prediction: J')
plt.plot(xs, y_vars, label='Variance: V')
plt.plot(xs, ff_var(xs)[0], label='FF-Variance Prediction: V')
plt.scatter(tf.squeeze(initial_query_points), forrester_func(initial_query_points), label='Initial Point')
plt.scatter(tf.squeeze(data_query_points[num_initial_points:, 0]), 
            forrester_func(data_query_points[num_initial_points:, 0]), 
            c = tf.range(data_query_points.shape[0] - num_initial_points), label='RBO Added Point', cmap='viridis')
plt.xlabel('Input X')
plt.legend()
plt.show()

# %% [markdown]
# In $\mathbb{F}_{MV}$ space:

# %%
plt.figure()
plt.scatter(Forrester().fmean_objective(tf.expand_dims(base_sample, -2))(data_query_points[num_initial_points:]), 
            Forrester().fvar_objective(tf.expand_dims(base_sample, -2))(data_query_points[num_initial_points:]), 
            c = tf.range(data_query_points.shape[0] - num_initial_points), 
            label='RBO Added Points', cmap='viridis')
plt.scatter(mean_var_pf[:, 0], mean_var_pf[:, 1], s=5, marker='X', label='Reference Pareto Front', color='r')
plt.xlabel('Mean $\mathbb{J}$')
plt.ylabel('Variance $\mathbb{V}$')
plt.legend()
plt.show()
