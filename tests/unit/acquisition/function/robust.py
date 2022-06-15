import numpy as np
import tensorflow as tf

# %%
from trieste.acquisition.function.robust import NoisyMinValueEntropySearchVarianceConstraint

# %% [markdown]
# # Noise-free optimization with Expected Improvement


np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Describe the problem
# In this example, we look to find the minimum value of the two-dimensional Branin function over the hypercube $[0, 1]^2$. We can represent the search space using a `Box`, and plot contours of the Branin over this space.
#
#

from trieste.acquisition.rule import EfficientGlobalOptimization

# %%
from trieste.objectives import SCALED_BRANIN_MINIMUM, scaled_branin
from trieste.objectives.utils import mk_observer
from trieste.space import Box

search_space = Box([0, 0], [1, 1])

# %% [markdown]
# ## Sample the observer over the search space
#
# Sometimes we don't have direct access to the objective function. We only have an observer that indirectly observes it. In _Trieste_, an observer can output a number of datasets. In our case, we only have one dataset, the objective. We can convert a function with `branin`'s signature to a single-output observer using `mk_observer`.
#
# The optimization procedure will benefit from having some starting data from the objective function to base its search on. We sample a five point space-filling design from the search space and evaluate it with the observer. For continuous search spaces, Trieste supports random, Sobol and Halton initial designs.

# %%
import trieste

observer = trieste.objectives.utils.mk_observer(scaled_branin)

num_initial_points = 20
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

# %% [markdown]
# ## Model the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use Gaussian Process (GP) regression for this, as provided by GPflow. The model will need to be trained on each step as more points are evaluated, so we'll package it with GPflow's Scipy optimizer.
#
# We put priors on the parameters of our GP model's kernel in order to stabilize model fitting. We found the priors below to be highly effective for objective functions defined over the unit hypercube and with an ouput standardized to have zero mean and unit variance. For objective functions with different scaling, other priors will likely be more appropriate. Our fitted model uses the maximum a posteriori estimate of these kernel parameters, as found by optimizing the kernel parameters starting from the best of `num_kernel_samples` random samples from the kernel parameter priors.
#
# If we do not specify kernel priors, then Trieste returns the maximum likelihood estimate of the kernel parameters.

# %%
import gpflow
import tensorflow_probability as tfp

from trieste.models.gpflow import GPflowModelConfig


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance, lengthscales=[0.2, 0.2])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.cast(-2.0, dtype=tf.float64), prior_scale
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.lengthscales), prior_scale
    )
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GPflowModelConfig(
        **{
            "model": gpr,
            "model_args": {
                "num_kernel_samples": 100,
            },
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)},
            },
        }
    )


model = build_model(initial_data)

# %% [markdown]
# ## Run the optimization loop
#
# We can now run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its `optimize` method.
#
# The optimizer uses an acquisition rule to choose where in the search space to try on each optimization step. We'll use the default acquisition rule, which is Efficient Global Optimization with Expected Improvement.
#
# We'll run the optimizer for fifteen steps.
#
# The optimization loop catches errors so as not to lose progress, which means the optimization loop might not complete and the data from the last step may not exist. Here we'll handle this crudely by asking for the data regardless, using `.try_get_final_datasets()`, which will re-raise the error if one did occur. For a review of how to handle errors systematically, there is a [dedicated tutorial](recovering_from_errors.ipynb).

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

nmes_cv = NoisyMinValueEntropySearchVarianceConstraint(
    search_space,
    variance_constraint=tf.constant([2.0], dtype=tf.float64),
    noise_type="normal",
    noise_param=tf.constant([[1e-2, 0.0], [0.0, 1e-2]], dtype=tf.float64),
    rej_samples=100,
    grid_size=0,
    num_fourier_features=512,
    ff_method="RFF",
)
rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=nmes_cv)


num_steps = 15
result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)
dataset = result.try_get_final_dataset()
