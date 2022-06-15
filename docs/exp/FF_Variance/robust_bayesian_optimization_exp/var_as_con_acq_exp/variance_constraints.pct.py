# %% [markdown]
# # Variance as Inequality constraints

# %%
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow_probability import distributions as tfd

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## 1D SinLinear Function


# %% [markdown]
# Lets first define the noise parameter:

# %% [markdown]
# 1. Normal noise

# %%
noise_type='normal'
noise_param = tf.constant([[0.001]], dtype=tf.float64)  # Variance

noise_dist = tfd.Normal(loc = [0.0], scale = tf.sqrt(noise_param[0]))

# %% [markdown]
# Uniform Noise

# %%
noise_type='uniform'
noise_param = tf.constant([0.05], dtype=tf.float64)  # Variance

noise_dist = tfd.Uniform(-noise_param, noise_param)

# %% [markdown]
# -----------
#
# import numpy as np
# from matplotlib import pyplot as plt

# %%
import trieste
from trieste.bayesian_optimizer import EfficientGlobalOptimization
from trieste.objectives.single_objectives import SinLinear
from trieste.space import Box

# %%
sinlinear = SinLinear().objective()
observer = trieste.objectives.utils.mk_observer(sinlinear)

# %%
search_space = Box(*SinLinear.bounds)
num_objective = 1

# %% [markdown]
# Lets take an initial visualization of the objective function, as well as its variance (constraint function)

# %%
# %matplotlib notebook

# %%
xs = tf.linspace([0], [1], 100)
ys = sinlinear(xs)


def sinlinear_robust(input):
    input_dists = tf.expand_dims(input, -2) + tf.cast(
        noise_dist.sample(20000), dtype=xs.dtype)  # [N, mc, 1]
    noisy_res = sinlinear(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return tf.concat([y_means, y_vars], -1)

y_means, y_vars = tf.split(sinlinear_robust(xs), 2, axis=-1)

plt.figure()
plt.plot(xs, ys, label='function')
plt.plot(xs, y_means, label='Mean: J')
plt.plot(xs, y_vars, label='Variance: V')
plt.hlines(0.14, 0.0, 1.0, label='Variance Constraint')
plt.legend()
plt.show()

# %% [markdown]
# Calculate Analytical Min Value

# %%
base_sample = tf.cast(noise_dist.sample(10000000), dtype=xs.dtype)
def penalized_obj(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = sinlinear(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return tf.squeeze(y_means + 1000* tf.maximum(y_vars - 0.14, 0.0)).numpy()


# %%
penalized_obj(tf.constant([[0.4]], dtype=tf.float64))

# %%
from scipy.optimize import minimize

res = minimize(penalized_obj, tf.constant([[0.7]], dtype=tf.float64))

# %%
res

# %%
np.savetxt('SinLinear_MV_Uniform_0.05_Opt_X.txt', res.x)
np.savetxt('SinLinear_MV_Uniform_0.05_Opt_F.txt', np.atleast_1d(res.fun))

# %% [markdown]
# Lets' calculate the worst mean obj for utility gap metric

# %%
from scipy.optimize import Bounds


def single_obj(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = sinlinear(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)

    return tf.squeeze(y_means).numpy()

worst_res = minimize(lambda at: -single_obj(at), tf.constant([[0.5]], dtype=tf.float64),
               bounds=Bounds(0.0, 1.0))

# %%
worst_res

# %% [markdown]
# The reference point is:

# %% [markdown]
# Let's randomly sample some initial data from the observer ...

# %%
num_initial_points = 5
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %%
import gpflow

from trieste.models.gpflow.models import GaussianProcessRegression


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance, lengthscales=[1.0])
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


model = build_model(initial_data)

# %% [markdown]
# Now we start optimization:

# %%
from trieste.acquisition.function.robust import FF_MV_qECI

ff_mva_ehvi = FF_MV_qECI(noise_type=noise_type, noise_param = noise_param, variance_threshold=0.14,
                          ff_method='QFF', opt_ff_num=128, mc_num=128, infer_mc_num=10000, implicit_sample=True)
rule: EfficientGlobalOptimization = EfficientGlobalOptimization(
    builder=ff_mva_ehvi, num_query_points=1, num_initial_samples=100, num_optimization_runs=2)

# %%
num_steps = 20
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)

# %%
dataset = result.try_get_final_dataset()
data_query_points = dataset.query_points
data_observations = dataset.observations

# Plot in design space
from trieste.acquisition.sampler import QFFVarianceSampler

sampler = QFFVarianceSampler(
    noise_type, model, dataset, num_features=128, noise=noise_param
)
ff_mean = sampler.get_mean_trajectory(get_mean=True)
ff_var = sampler.get_var_trajectory(get_mean=True)
plt.figure()
plt.plot(xs, ys, label='function')
plt.plot(xs, y_means, label='Mean: J')
plt.plot(xs, ff_mean(xs)[0], label='FF-Mean: J')
plt.plot(xs, y_vars, label='Variance: V')
plt.plot(xs, ff_var(xs)[0], label='FF-Variance: V')
plt.scatter(tf.squeeze(initial_query_points), sinlinear(initial_query_points), label='Initial Point')
plt.scatter(tf.squeeze(data_query_points[num_initial_points:, 0]), sinlinear(data_query_points[num_initial_points:, 0]), label='Added Point')
plt.xlabel('Input X')
plt.legend()
plt.show()

# %% [markdown]
# -------------

# %% [markdown]
# ## 1D Forrester Function

# %% [markdown]
# Lets first define the noise parameter:

# %% [markdown]
# 1. Normal noise

# %%
noise_type='normal'
noise_param = tf.constant([[0.005]], dtype=tf.float64)  # Variance

noise_dist = tfd.Normal(loc = [0.0], scale = tf.sqrt(noise_param[0]))

import numpy as np
from matplotlib import pyplot as plt

# %% [markdown]
# Uniform Noise

# %%
noise_type='uniform'
noise_param = tf.constant([0.1], dtype=tf.float64)  # Variance

noise_dist = tfd.Uniform(-noise_param, noise_param)

import numpy as np
from matplotlib import pyplot as plt

# %%
import trieste
from trieste.bayesian_optimizer import EfficientGlobalOptimization
from trieste.objectives.single_objectives import Forrester
from trieste.space import Box

# %% [markdown]
# ----------


# %%
forrester_func = Forrester().objective()
observer = trieste.objectives.utils.mk_observer(forrester_func)

# %%
search_space = Box(*Forrester.bounds)
num_objective = 1

# %% [markdown]
# Lets take an initial visualization of the objective function, as well as its mean and variance

# %%
# %matplotlib notebook

# %%
xs = tf.linspace([0], [1], 100)
ys = forrester_func(xs)


def forrester_robust(input):
    input_dists = tf.expand_dims(input, -2) + tf.cast(
        noise_dist.sample(100000), dtype=xs.dtype)  # [N, mc, 1]
    noisy_res = forrester_func(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return tf.concat([y_means, y_vars], -1)

y_means, y_vars = tf.split(forrester_robust(xs), 2, axis=-1)

plt.figure()
plt.plot(xs, ys, label='function')
plt.plot(xs, y_means, label='Mean: J')
plt.plot(xs, y_vars, label='Variance: V')
plt.hlines(1, 0.0, 1.0)
plt.legend()
plt.show()

# %% [markdown]
# Calculate Analytical Min Value

# %%
base_sample = tf.cast(noise_dist.sample(1000000), dtype=xs.dtype)


# %%
def single_obj(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = forrester_func(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)

    return tf.squeeze(y_means).numpy()

def cons(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = forrester_func(input_dists) # [N, mc, 1]
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return -tf.squeeze(y_vars - 1.0).numpy()


# %%
single_obj(tf.constant([[0.3]], dtype=tf.float64))

# %%
from scipy.optimize import Bounds, minimize

res = minimize(single_obj, tf.constant([[0.168]], dtype=tf.float64), constraints = {'type':'ineq', 'fun': cons},
               bounds=Bounds(0.0, 1.0))

# %%
res

# %%
single_obj(res.x)

# %%
cons(res.x)

# %%
single_obj(np.array([1.853863902550537679e-01]))

# %% [markdown]
# Lets' calculate the worst mean obj for utility gap metric

# %%
worst_res = minimize(lambda at: -single_obj(at), tf.constant([[0.4]], dtype=tf.float64),
               bounds=Bounds(0.0, 1.0))

# %%
worst_res

# %% [markdown]
# Normal

# %%
np.savetxt('Forrester_MV_Normal_0.005_Opt_X.txt', res.x)
np.savetxt('Forrester_MV_Normal_0.005_Opt_F.txt', np.atleast_1d(res.fun) - 5e-3)

# %% [markdown]
# Uniform

# %%
np.savetxt('Forrester_MV_Uniform_0.1_Opt_X.txt', res.x)
np.savetxt('Forrester_MV_Uniform_0.1_Opt_F.txt', np.atleast_1d(res.fun) - 5e-3)

# %% [markdown]
# Let's randomly sample some initial data from the observer ...

# %%
num_initial_points = 5
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %%
import gpflow

from trieste.models.gpflow.models import GaussianProcessRegression


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance, lengthscales=[1.0])
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


model = build_model(initial_data)

# %% [markdown]
# Now we start optimization:

# %%
from trieste.acquisition.function.robust import FF_MV_qECI

ff_mva_ehvi = FF_MV_qECI(noise_type=noise_type, noise_param = noise_param, variance_threshold = 2.0, 
                          ff_method='QFF', opt_ff_num=128, mc_num=64, infer_mc_num=10000)
rule: EfficientGlobalOptimization = EfficientGlobalOptimization(
    builder=ff_mva_ehvi, num_query_points=1, num_initial_samples=100, num_optimization_runs=1)

# %%
num_steps = 20
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)

# %%
# %matplotlib notebook

# %%
dataset = result.try_get_final_dataset()
data_query_points = dataset.query_points
data_observations = dataset.observations

# Plot in design space
from trieste.acquisition.sampler import QFFVarianceSampler

sampler = QFFVarianceSampler(
    noise_type, model, dataset, num_features=128, noise=noise_param
)
ff_mean = sampler.get_mean_trajectory(get_mean=True)
ff_var = sampler.get_var_trajectory(get_mean=True)
plt.figure()
plt.plot(xs, ys, label='function')
plt.plot(xs, y_means, label='Mean: J')
plt.plot(xs, ff_mean(xs)[0], label='FF-Mean: J')
plt.plot(xs, y_vars, label='Variance: V')
plt.plot(xs, ff_var(xs)[0], label='FF-Variance: V')
plt.scatter(tf.squeeze(initial_query_points), forrester_func(initial_query_points), label='Initial Point')
plt.scatter(tf.squeeze(data_query_points[num_initial_points:, 0]), forrester_func(data_query_points[num_initial_points:, 0]),
            c = np.arange(data_query_points.shape[0] - num_initial_points), cmap='viridis', label='Added Point')
plt.xlabel('Input X')
plt.legend()
plt.show()

# %% [markdown]
# --------------

# %% [markdown]
# ## 2D Bird Function

# %% [markdown]
# 1. Normal noise

# %%
noise_type='normal'
noise_param = tf.constant([[0.02, 0.0], [0.0, 0.02]], dtype=tf.float64)  # Variance

noise_dist = tfd.MultivariateNormalFullCovariance(loc = [0.0, 0.0], covariance_matrix= noise_param)

# %% [markdown]
# -----------
#
# import numpy as np
# from matplotlib import pyplot as plt

# %%
import trieste
from trieste.bayesian_optimizer import EfficientGlobalOptimization
from trieste.objectives.single_objectives import Bird
from trieste.space import Box

# %%
bird_func = Bird().objective()
observer = trieste.objectives.utils.mk_observer(bird_func)

# %%
search_space = Box(*Bird.bounds)
num_objective = 1

# %% [markdown]
# Lets take an initial visualization of the objective function, as well as its mean and variance
#
# from PyOptimize.utils.visualization import view_2D_function_in_contour

# %%
from docs.notebooks.util.plotting import plot_function_2d
from docs.notebooks.util.plotting_plotly import plot_function_plotly

# %% [markdown]
# ### Visualize

# %%
plot_function_plotly(bird_func, *Bird.bounds, grid_density=100)

# %%
# %matplotlib notebook

# %%
from PyOptimize.utils.visualization import view_2D_function_in_contour

plt.figure()
view_2D_function_in_contour(bird_func, [[0, 1]] * 2, show=True, colorbar=True, plot_fidelity=64)

# %%
plot_function_plotly(Bird().fmean_objective(tf.expand_dims(noise_dist.sample(1000), -2)), *Bird.bounds, grid_density=100)

# %%
plt.figure()
plt_inst = view_2D_function_in_contour(Bird().fmean_objective(tf.expand_dims(noise_dist.sample(1000), -2)), [[0, 1]] * 2, 
                                       show=False, colorbar=True, plot_fidelity=64, levels=10)
plt_inst.title('Mean Plot')

# %%
plt.figure()
plt_inst = view_2D_function_in_contour(Bird().fvar_objective(tf.expand_dims(noise_dist.sample(2000), -2)), 
                                       [[0, 1]] * 2, show=False, colorbar=True, plot_fidelity=64, levels=15)
plt_inst.title('Var Plot')


# %% [markdown]
# We try to optimize using NSGA2

# %%
def bird_robust(input):
    input_dists = tf.expand_dims(input, -2) + tf.cast(
        noise_dist.sample(10000), dtype=input.dtype)  # [N, mc, 1]
    noisy_res = bird_func(input_dists) # [N, mc, 1]
    y_means = tf.reduce_mean(noisy_res, -2)
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return  tf.concat([y_means, y_vars], -1)


# %% [markdown]
# Now we start performing RBO

# %%
num_initial_points = 100
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %%
import gpflow

from trieste.models.gpflow.models import GaussianProcessRegression


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance, lengthscales=[1.0] * 2)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


model = build_model(initial_data)

# %%
model.optimize(initial_data)

# %% [markdown]
# Now we start optimization:

# %%
from trieste.acquisition.function.robust import FF_MV_qECI

ff_mva_ehvi = FF_MV_qECI(noise_type=noise_type, noise_param = noise_param,variance_threshold = 200.0, 
                          ff_method='QFF', opt_ff_num=30, mc_num=64, infer_mc_num=5000, max_batch_element=20)
rule: EfficientGlobalOptimization = EfficientGlobalOptimization(
    builder=ff_mva_ehvi, num_query_points=1, num_initial_samples=20, num_optimization_runs=1)

# %%
num_steps = 30
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)

# %%
# %matplotlib notebook

# %%
from PyOptimize.utils.visualization import view_2D_function_in_contour

dataset = result.try_get_final_dataset()
data_query_points = dataset.query_points
data_observations = dataset.observations

plt.figure()
plt_inst  = view_2D_function_in_contour(bird_func, [[0, 1]] * 2, show=False, colorbar=True, plot_fidelity=64)
plt_inst.scatter(dataset.query_points[:num_initial_points, 0], dataset.query_points[:num_initial_points, 1], label='Init')
plt_inst.scatter(dataset.query_points[num_initial_points:, 0], dataset.query_points[num_initial_points:, 1], 
                c = np.arange(data_query_points.shape[0] - num_initial_points), cmap='viridis', label='Added Point')
plt_inst.legend()
plt_inst.show()

# %% [markdown]
# We have to have a look at the Variance Inference seeing if it is accurate

# %%
from trieste.acquisition.sampler import QFFVarianceSampler

sampler = QFFVarianceSampler(
    noise_type, model, initial_data, num_features=30, noise=noise_param
)
ff_var = sampler.get_var_trajectory(get_mean=True, max_batch_element_num=50)

# %%
plt.figure()
plt_inst = view_2D_function_in_contour(ff_var, [[0, 1]] * 2, show=False, colorbar=True, plot_fidelity=64, levels=15)
plt_inst.title('Var Plot')

# %% [markdown]
# ## 2D Branin Function

# %% [markdown]
# 1. Normal noise

# %%
noise_type='normal'
noise_param = tf.constant([[0.01,0.0], [0.0, 0.01]], dtype=tf.float64)  # Variance

noise_dist = tfd.MultivariateNormalFullCovariance(loc = [0.0, 0.0], covariance_matrix= noise_param)

# %% [markdown]
# Uniform noise

# %%
noise_type='uniform'
noise_param = tf.constant([0.05, 0.05], dtype=tf.float64)  # Variance

noise_dist = tfd.Uniform(-noise_param, noise_param)

# %% [markdown]
# -----------
#
# import numpy as np
# from matplotlib import pyplot as plt

# %%
import trieste
from trieste.bayesian_optimizer import EfficientGlobalOptimization
from trieste.objectives.single_objectives import Branin
from trieste.space import Box

# %%
branin_func = Branin().objective()
observer = trieste.objectives.utils.mk_observer(branin_func)

# %%
search_space = Box(*Branin.bounds)
num_objective = 2

# %% [markdown]
# Lets take an initial visualization of the objective function, as well as its mean and variance
#
#

# %%
from PyOptimize.utils.visualization import view_2D_function_in_contour

# %%
from docs.notebooks.util.plotting import plot_function_2d
from docs.notebooks.util.plotting_plotly import plot_function_plotly

# %% [markdown]
# -----------------

# %% [markdown]
# Calculate Analytical Min Value

# %%
base_sample = tf.cast(noise_dist.sample(100000), dtype=tf.float64)


# %%
def single_obj(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = branin_func(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)

    return tf.squeeze(y_means).numpy()

def cons(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = branin_func(input_dists) # [N, mc, 1]
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return -tf.squeeze(y_vars - 160.0).numpy()


# %%
from scipy.optimize import Bounds, minimize

res = minimize(single_obj, tf.constant([[0.7, 0.09]], dtype=tf.float64), constraints = {'type':'ineq', 'fun': cons},
               bounds=Bounds([0.0, 0.0], [1.0, 1.0]))

# %%
res

# %% [markdown]
# Normal

# %%
np.savetxt('Branin_MV_Normal_0.01_Opt_X.txt', res.x)
np.savetxt('Branin_MV_Normal_0.01_Opt_F.txt', np.atleast_1d(res.fun))

# %% [markdown]
# Uniform

# %%
np.savetxt('Branin_MV_Uniform_0.05_Opt_X.txt', res.x)
np.savetxt('Branin_MV_Uniform_0.05_Opt_F.txt', np.atleast_1d(res.fun))

# %% [markdown]
# --------

# %%
from scipy.optimize import Bounds


def single_obj(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = branin_func(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)

    return tf.squeeze(y_means).numpy()

worst_res = minimize(lambda at: -single_obj(at), tf.constant([[0.2, 0.2]], dtype=tf.float64),
               bounds=Bounds((0.0, 0.0), (1.0, 1.0)))

# %%
worst_res

# %% [markdown]
# ### Visualize

# %%
plot_function_plotly(branin_func, *Branin.bounds, grid_density=100)

# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
from PyOptimize.utils.visualization import view_2D_function_in_contour

plt.figure()
view_2D_function_in_contour(branin_func, [[0, 1]] * 2, show=True, colorbar=True, plot_fidelity=64)

# %%
plot_function_plotly(Branin().fmean_objective(tf.expand_dims(noise_dist.sample(1000), -2)), *Branin.bounds, grid_density=100)

# %%
plt.figure()
plt_inst = view_2D_function_in_contour(lambda at: Branin().fmean_objective(tf.expand_dims(noise_dist.sample(10000), -2))(at), [[0, 1]] * 2, show=False, colorbar=True, plot_fidelity=64)
plt_inst.title('Mean Plot')

# %%
# %matplotlib notebook

# %% [markdown]
# NOTE!!!!!!!!!!!!!!!!!!!!! We put a log here!!!!!!!!!!

# %%
tf.exp(4.0)

# %%
plt.figure()
plt_inst = view_2D_function_in_contour(lambda at: tf.math.log(Branin().fvar_objective(tf.expand_dims(noise_dist.sample(10000), -2))(at)), [[0, 1]] * 2, show=False, 
                                       colorbar=True, plot_fidelity=64, levels=20)
plt.scatter(*res.x)
plt_inst.title('Var Plot')


# %% [markdown]
# We try to optimize using NSGA2

# %%
def branin_robust(input):
    input_dists = tf.expand_dims(input, -2) + tf.cast(
        noise_dist.sample(10000), dtype=input.dtype)  # [N, mc, 1]
    noisy_res = branin_func(input_dists) # [N, mc, 1]
    y_means = tf.reduce_mean(noisy_res, -2)
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return  tf.concat([y_means, y_vars], -1)


# %% [markdown]
# It seems this problem can be solved, now let's start BO

# %%
num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %%
import gpflow

from trieste.models.gpflow.models import GaussianProcessRegression


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance, lengthscales=[1.0] * 2)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


model = build_model(initial_data)

# %% [markdown]
# Now we start optimization:

# %%
from trieste.acquisition.function.robust import FF_MV_qECI

ff_mva_ehvi = FF_MV_qECI(noise_type=noise_type, noise_param = noise_param, variance_threshold=55, 
                          ff_method='QFF', opt_ff_num=30, mc_num=128, infer_mc_num=10000, max_batch_element=20)
rule: EfficientGlobalOptimization = EfficientGlobalOptimization(
    builder=ff_mva_ehvi, num_query_points=1, num_initial_samples=20, num_optimization_runs=1)

# %%
num_steps = 40
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)

# %%
# %matplotlib notebook

# %% [markdown]
# Now we visualize the result

# %%
from PyOptimize.utils.visualization import view_2D_function_in_contour

dataset = result.try_get_final_dataset()
data_query_points = dataset.query_points
data_observations = dataset.observations

plt.figure()
plt_inst  = view_2D_function_in_contour(branin_func, [[0, 1]] * 2, show=False, colorbar=True, plot_fidelity=64)
plt_inst.scatter(dataset.query_points[:num_initial_points, 0], dataset.query_points[:num_initial_points, 1], label='Init')
plt_inst.scatter(dataset.query_points[num_initial_points:, 0], dataset.query_points[num_initial_points:, 1], label='FF-MV-qEHVI',
                c = np.arange(dataset.query_points.shape[0] - 10), cmap='viridis')
plt_inst.legend()
plt_inst.show()

# %% [markdown]
# ## 3D Hartmann3

# %% [markdown]
# 1. Normal noise

# %%
noise_type='normal'
noise_param = tf.constant([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=tf.float64)  # Variance

noise_dist = tfd.MultivariateNormalFullCovariance(loc = [0.0, 0.0, 0.0], covariance_matrix= noise_param)

# %% [markdown]
# Uniform
#

# %%
noise_type='uniform'
noise_param = tf.constant([0.05, 0.05, 0.05], dtype=tf.float64)  # Variance
# noise_param = tf.constant([0.1, 0.1, 0.1], dtype=tf.float64)
noise_dist = tfd.Uniform(-noise_param, noise_param)

# %% [markdown]
# ----------

# %%
import numpy as np
from matplotlib import pyplot as plt

# %%
import trieste
from trieste.bayesian_optimizer import EfficientGlobalOptimization
from trieste.objectives.single_objectives import Hartmann_3
from trieste.space import Box

# %%
hartmann3_func = Hartmann_3().objective()
observer = trieste.objectives.utils.mk_observer(hartmann3_func)

# %%
search_space = Box(*Hartmann_3.bounds)
num_objective = 2

# %% [markdown]
# Calculate Analytical Min Value

# %%
base_sample = tf.cast(noise_dist.sample(10000), dtype=tf.float64)


# %%
def single_obj(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = hartmann3_func(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)

    return tf.squeeze(y_means).numpy()

def cons(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = hartmann3_func(input_dists) # [N, mc, 1]
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return tf.squeeze(0.12 - y_vars).numpy() # 0.15 for normal!!!!!


# %%
from scipy.optimize import Bounds, minimize

res = minimize(single_obj, tf.constant([[0.2, 0.5, 0.5]], dtype=tf.float64), constraints = {'type':'ineq', 'fun': cons},
               bounds=Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))

# %%
res

# %% [markdown]
# We also investigate wether it is an active constraint

# %%
unc_res = minimize(single_obj, tf.constant([[0.2, 0.5, 0.5]], dtype=tf.float64), bounds=Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))

# %%
unc_res

# %% [markdown]
# Normal

# %%
np.savetxt('Hartmann3_MV_Normal_0.01_Opt_X.txt', res.x)
np.savetxt('Hartmann3_Normal_0.01_Opt_F.txt', np.atleast_1d(res.fun))

# %% [markdown]
# Uniform

# %%
np.savetxt('Hartmann3_MV_Uniform_0.1_Opt_X.txt', res.x)
np.savetxt('Hartmann3_Uniform_0.1_Opt_F.txt', np.atleast_1d(res.fun))

# %% [markdown]
# ---------------

# %%

from scipy.optimize import Bounds, minimize


def single_obj(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = hartmann3_func(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)

    return tf.squeeze(y_means).numpy()

worst_res = minimize(lambda at: -single_obj(at), tf.constant([[0.1, 0.1, 0.1]], dtype=tf.float64),
               bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))

# %%
worst_res

# %% [markdown]
# We visualize the feasible region, seeing if it is sufficiently large (easy for opt)

# %%
xs = np.random.uniform(size=(1000, 3))
cons_res = cons(xs)

# %%
# %matplotlib notebook

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs[cons_res>0][:, 0], xs[cons_res>0][:, 1], xs[cons_res>0][:, 2], c = 'b', marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

# %% [markdown]
# ## 3D Robot Pushing
#

# %% [markdown]
# 1. Normal noise

# %%
noise_type='normal'
noise_param = tf.constant([[0.0004, 0.0, 0.0], [0.0, 0.0004, 0.0], [0.0, 0.0, 7.13436385255648e-05]], dtype=tf.float64)  # Variance

noise_dist = tfd.MultivariateNormalFullCovariance(loc = [0.0, 0.0, 0.0], covariance_matrix= noise_param)

# %% [markdown]
# Uniform

# %%
noise_type='uniform'
noise_param = tf.constant([0.03, 0.03, 0.01], dtype=tf.float64)  # Variance

noise_dist = tfd.Uniform(-noise_param, noise_param)

# %% [markdown]
# ----------

# %%
import numpy as np
from matplotlib import pyplot as plt

# %%
import trieste
from trieste.bayesian_optimizer import EfficientGlobalOptimization
from trieste.objectives.single_objectives import Robot_Pushing_3D
from trieste.space import Box

# %%
# rp_func = Robot_Pushing_3D(2, 2).objective()
rp_func = Robot_Pushing_3D(4, 3).objective()
observer = trieste.objectives.utils.mk_observer(rp_func)

# %%
rp_func(tf.constant([[0.0, 0.0 , 0.0]]))

# %%
search_space = Box(*Robot_Pushing_3D.bounds)

# %% [markdown]
# Calculate Analytical Min Value

# %%
base_sample = tf.cast(noise_dist.sample(200), dtype=tf.float64)


# %%
def single_obj(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    # print(input_dists.shape)
    noisy_res = rp_func(input_dists) # [N, mc, 1]

    y_means = tf.reduce_mean(noisy_res, -2)

    return tf.squeeze(y_means).numpy()

def cons(at):
    input_dists = tf.expand_dims(at, -2) + base_sample  # [N, mc, 1]
    noisy_res = rp_func(input_dists) # [N, mc, 1]
    y_vars = tf.math.reduce_variance(noisy_res, -2)
    return tf.squeeze(0.03 - y_vars).numpy() # we use this for uniform 
    # return tf.squeeze(0.05 - y_vars).numpy() # we use this for normal 
    # return tf.squeeze(0.00 - y_vars).numpy()


# %% [markdown]
# We 1st taste unconstraint opt

# %%
from scipy.optimize import Bounds, minimize

# %%
from scipy.optimize import Bounds, minimize
# 5.917137007679790922e-02, 1.678041732391259977e-01, 8.509166636115992333e-01
# res = minimize(single_obj, tf.constant([[0.21599655, 0.21647106, 0.5  ]], dtype=tf.float64), bounds=Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
res = minimize(single_obj, tf.constant([[5.917137007679790922e-02, 1.678041732391259977e-01, 8.509166636115992333e-01  ]], dtype=tf.float64), bounds=Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))


# %%
res

# %%
mean_opt_x = res.x
mean_opt_f = res.fun

# %%
single_obj(res.x)

# %%
cons(res.x)

# %%
for i in range(30):
    rec_x = np.loadtxt(rf'C:\Users\Administrator\Desktop\trieste_mro_dev\docs\exp\FF_Variance\robust_bayesian_optimization_exp\var_as_con_acq_exp\exp_res\Robot_Pushing_3D\FF-MV-qECI\uniform\q1_rff\Robot_Pushing_3D_{i}_In_Sample_recommend_input_q1_.txt')
    print(f'obj: {single_obj(rec_x)}')
    print(f'cons: {cons(rec_x)}')

# %% [markdown]
# ------------------

# %%
noise_free_res = minimize(lambda at: tf.squeeze(rp_func(at)), tf.constant([[0.21599655, 0.21647106, 0.5 ]], dtype=tf.float64), 
                          bounds=Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))


# %%
noise_free_res

# %%
rp_func(tf.constant([[0.35446959, 0.38946343, 0.5  ]], dtype=tf.float64))

# %%
cons(noise_free_res.x)

# %% [markdown]
# Now we calculate the variance as constraint case

# %%
# 0.22508935, 0.22557234, 0.5
# 0.21599655, 0.21647106, 0.5
# 6.904466905926164022e-02, 1.625222006556177601e-01, 8.418933754813829884e-01
cons_res = minimize(single_obj, tf.constant([[ 1.492123614039568003e-01, 2.231424077007531925e-01, 8.087948593956592047e-01  ]], dtype=tf.float64), constraints = {'type':'ineq', 'fun': cons},
              bounds=Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))

# %%
1.334319991895374091e+00 - 0.34216346960189675

# %%
cons_res

# %%
cons(cons_res.x)

# %% [markdown]
# We test BO found:

# %%
single_obj(tf.constant([0.000000000000000000e+00, 1.149742522232360042e-03, 8.922492549828922037e-01], dtype=tf.float64))

# %%
cons(tf.constant([0.000000000000000000e+00, 1.149742522232360042e-03, 8.922492549828922037e-01], dtype=tf.float64))

# %% [markdown]
# We visualize the feasible region, seeing if it is sufficiently large (easy for opt)

# %%
xs = np.random.uniform(size=(1000, 3))

# %%
cons_res = [cons(x) for x in xs]

# %%
# %matplotlib notebook

# %%
cons_res = np.asarray(cons_res)

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs[cons_res>0][:, 0], xs[cons_res>0][:, 1], xs[cons_res>0][:, 2], c = 'b', marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

# %% [markdown]
# Now we calculate the worst obj

# %%
from scipy.optimize import Bounds, minimize

res = minimize(lambda at: -single_obj(at), tf.constant([[0.21599655, 0.21647106, 0.5 ]], dtype=tf.float64), bounds=Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))

# %%
res

# %%
np.savetxt('Robot_Pushing_3D_Normal_Opt_F.txt', np.atleast_1d(cons_res.fun))

# %%
np.savetxt('Robot_Pushing_3D_Uniform_Opt_F.txt', np.atleast_1d(cons_res.fun))

# %% [markdown]
# ------------

# %%
print(noise_free_res)

# %%
cons(noise_free_res.x)

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
