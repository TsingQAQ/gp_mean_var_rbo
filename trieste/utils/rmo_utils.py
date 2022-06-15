from collections.abc import Callable

import tensorflow as tf
import tensorflow_probability as tfp

from ..acquisition.multi_objective import Pareto, non_dominated
from ..types import TensorType

# from ..bayesian_optimizer import OptimizationResult


def estimate_fmean(x, f, samples):
    """
    estimate fmean on point x based on MC sampling
    """
    tf.debugging.assert_shapes([(samples, ("N", 1, "D"))])
    Xs = tf.tile(x[tf.newaxis, ...], [samples.shape[0], 1, 1]) + tf.cast(samples, dtype=x.dtype)
    fs = tf.vectorized_map(f, Xs)
    return tf.reduce_mean(fs, axis=0)


def estimate_fvar(x, f, samples):
    """
    estimate fmean on point x based on MC sampling
    """
    tf.debugging.assert_shapes([(samples, ("N", 1, "D"))])
    Xs = tf.tile(x[tf.newaxis, ...], [samples.shape[0], 1, 1]) + tf.cast(samples, dtype=x.dtype)
    fs = tf.vectorized_map(f, Xs)
    return tf.math.reduce_variance(fs, axis=0)


def estimate_fmean_and_fvar(x, f, samples):
    """
    estimate fmean on point x based on MC sampling
    """
    tf.debugging.assert_shapes([(samples, ("N", 1, "D"))])
    Xs = tf.tile(x[tf.newaxis, ...], [samples.shape[0], 1, 1]) + tf.cast(samples, dtype=x.dtype)
    try:
        fs = tf.vectorized_map(f, Xs)
    except:
        # fs = tf.map_fn(f, Xs)
        fs = tf.convert_to_tensor([f(x) for x in Xs], dtype=Xs.dtype)
    return tf.concat([tf.math.reduce_mean(fs, axis=0), tf.math.reduce_variance(fs, axis=0)], -1)


def gen_predict_pertubated_dataset(x, models, noise, x_mc_num=10000, f_mc_num=100):
    """
    use existing datasets to generate robust datases based on model prediction:
    y = \int p(y|x+zeta) d zeta
    """
    X = x
    Xs = tfp.distributions.Normal(X, scale=noise).sample(x_mc_num)
    n_fs = []
    for model_tag in models:
        n_fs.append(models[model_tag].model.predict_f_samples(Xs, f_mc_num, full_cov=False))
    Fs = tf.concat(n_fs, axis=-1)  # [X_MC * Y_MC * num * obj_num]
    return tf.reshape(Fs, shape=(Fs.shape[0] * Fs.shape[1] * Fs.shape[2], Fs.shape[3]))


def get_query_points_fmean(
    query_points: TensorType, f: Callable, noise_seed_samples: TensorType
) -> TensorType:
    """
    use query_points to generate datases with observation from function averaging
    every query points is independent
    :param query_points
    :param f: callable blackbox function
    :param noise_seed_samples
    """
    tf.debugging.assert_shapes([(query_points, ("N", "dim"))])
    tf.debugging.assert_shapes([(noise_seed_samples, ("MC_num", query_points.shape[-1]))])
    xs = tf.tile(tf.expand_dims(query_points, axis=0), [noise_seed_samples.shape[0], 1, 1])
    xs = xs + tf.expand_dims(noise_seed_samples, axis=1)  # [MC_num, N, dim]
    fmean = tf.reduce_mean(tf.vectorized_map(f, xs), axis=0)  # [N, dim]
    return fmean


def get_hv_dist_based_on_input_uncertainty(
    PFx: TensorType, f: Callable, noise_sample: TensorType, reference_pt
):
    """
    Get hypervolume distribution from input distribution samples
    """
    tf.debugging.assert_shapes([(noise_sample, ("mc_num", "N", "dim"))])
    tf.debugging.assert_shapes([(PFx, ("N", "dim"))])

    HV = []
    for noise in noise_sample:
        y_pertubated = f(PFx + noise)
        HV.append(Pareto(y_pertubated).hypervolume_indicator(reference_pt))
    return tf.concat(HV, axis=0)


def get_fmean_based_hv_history(
    result, fmean: Callable, ref_point: TensorType, index_tag: str
) -> list:
    """
    Get hypervolume from each iteration based on fmean function and specified reference point
    :param result
    :param fmean
    :param ref_point reference point in robust objective space
    """
    # assert isinstance(result, OptimizationResult)
    hv_hist = []
    for iter in range(len(result.history)):
        _model = result.history[iter].models
        _xs = result.history[iter].datasets[index_tag].query_points

        # get predicted pareto front corresponding input
        _observations = _model[index_tag].predict(_xs)[0]

        _, dominance = non_dominated(_observations)
        _pfx = tf.gather_nd(_xs.query_points, tf.where(tf.equal(dominance, 0)))

        # NOTE!!!, There is also a builtin re find of non-robust point
        _hv = Pareto(fmean(_pfx)).hypervolume_indicator(ref_point)
        hv_hist.append(_hv)
    return hv_hist
