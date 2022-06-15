# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: trieste_mro_dev
#     language: python
#     name: trieste_mro_dev
# ---

import gpflow
import matplotlib
import matplotlib.pyplot as plt

# %%
import numpy as np
import tensorflow as tf

from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.config import ModelConfig

matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}" r"\usepackage{amsfonts}"]
from trieste.objectives.single_objectives import GMM, Branin

# %%
from trieste.space import Box

# %%
ds = GMM.bounds
X = Box(*ds).sample(50)
# X = tf.concat([X, tf.constant([[0.74], [0.75]], dtype=X.dtype)], 0)
obj = lambda x: -GMM().objective()(x)
Y = obj(X)


# %%
single_obj_data = Dataset(X, Y)
variance = tf.math.reduce_variance(single_obj_data.observations)
kernel = gpflow.kernels.RBF(variance, lengthscales=[1.0] * 2)
# jitter = gpflow.kernels.White(1e-12)
gpr = gpflow.models.GPR(
    (single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1e-5
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

# %%
from trieste.acquisition.sampler import QFFVarianceSampler, RFFVarianceSampler
from trieste.data import Dataset

noise = tf.constant([[1e-3, 0.00], [0.00, 1e-3]], dtype=tf.float64)
rff_sampler = RFFVarianceSampler("normal", m2, Dataset(X, Y), num_features=16 ** 2, noise=noise)
qff_sampler = QFFVarianceSampler("normal", m2, Dataset(X, Y), num_features=16, noise=noise)


from util.plotting_plotly import plot_function_plotly

# fig, axs = plt.subplots(2, 3, figsize=(12, 5))

mean_gp = lambda xs: m2.predict(xs)[0]
fig = plot_function_plotly(mean_gp, Branin.bounds[0], Branin.bounds[1], grid_density=50)
fig.show()

# rff_mean = rff_sampler.get_mean_trajectory()
# fig = plot_function_plotly(rff_mean, Branin.bounds[0], Branin.bounds[1], grid_density=50)
# fig.show()
#
# qff_mean = qff_sampler.get_mean_trajectory()
# fig = plot_function_plotly(qff_mean, Branin.bounds[0], Branin.bounds[1], grid_density=50)
# fig.show()


rff_var = rff_sampler.get_var_trajectory(max_batch_element_num=66)
rff_var(tf.constant([[0.3, 0.5]], dtype=tf.float64))
fig = plot_function_plotly(
    lambda at: rff_var(at)[0], Branin.bounds[0], Branin.bounds[1], grid_density=50
)
fig.show()

qff_var = qff_sampler.get_var_trajectory(max_batch_element_num=20)
# qff_var(tf.constant([[0.3, 0.5]], dtype=tf.float64))
fig = plot_function_plotly(
    lambda at: qff_var(at)[0], Branin.bounds[0], Branin.bounds[1], grid_density=50
)
fig.show()
