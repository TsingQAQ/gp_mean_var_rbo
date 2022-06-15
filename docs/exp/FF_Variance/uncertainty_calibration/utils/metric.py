import tensorflow as tf
from gpflow.config import default_float


def log10_Wasserstein_distance(
    mean, covariance, approximate_mean, approximate_covariance, jitter=1e-12
):
    """
    Identify the decadic logarithm of the Wasserstein distance based on the means and covariance matrices.

    :param mean:The analytic mean, with a shape of [N*].
    :param covariance: The analytic covariance, with a shape of [N* x N*].
    :param approximate_mean: The approximate mean, with a shape of [N*].
    :param approximate_covariance: The approximate covariance, with a shape of [N* x N*].
    :param jitter: The jitter value for numerical robustness.

    :return: A scalar log distance value.
    """
    squared_mean_distance = tf.norm(mean - approximate_mean) ** 2
    square_root_covariance = tf.linalg.sqrtm(
        covariance + tf.eye(tf.shape(covariance)[0], dtype=covariance.dtype) * jitter
    )
    matrix_product = square_root_covariance @ approximate_covariance @ square_root_covariance
    square_root_matrix_product = tf.linalg.sqrtm(
        matrix_product + tf.eye(tf.shape(matrix_product)[0], dtype=matrix_product.dtype) * jitter
    )
    term = covariance + approximate_covariance - 2 * square_root_matrix_product
    trace = tf.linalg.trace(term)
    ws_distance = (squared_mean_distance + trace) ** 0.5
    log10_ws_distance = tf.math.log(ws_distance) / tf.math.log(
        tf.constant(10.0, dtype=default_float())
    )
    return log10_ws_distance
