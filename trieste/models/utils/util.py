"""
Copied from gpflow

"""
from typing import Optional

import tensorflow as tf
from gpflow.config import default_float, default_jitter
from gpflow.utilities.ops import leading_transpose


def modified_conditional(
    Kmn: tf.Tensor,
    Knm: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov=False,
    q_sqrt: Optional[tf.Tensor] = None,
    white=False,
):
    """
    A modified version of conditional
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2; 0, Kmm)

      p(g1) = N(g1; 0, Knn)
      p(g1 | g2) = N(g1; Knm (Kmm⁻¹) g2, Knn - Knm (Kmm⁻¹) Kmn)

    And
      q(g2) = N(g2; f, q_sqrt q_sqrtᵀ)

    This method computes the mean and (co)variance of
      q(g1) = ∫ q(g2) p(g1 | g2)

    :param Kmn: [..., MC_num, M, N]
    :param Kmm: [M, M]
    :param Knn: [..., MC_num, N, N]  or  [..., MC_num, N]
    :param f: [M, R]
    :param full_cov: bool
    :param q_sqrt: If this is a Tensor, it must have shape [R, M, M] (lower
        triangular) or [M, R] (diagonal)
    :param white: bool
    :return: [N, R]  or [R, N, N]
    """
    Lm = tf.linalg.cholesky(Kmm)
    return modified_conditional_with_lm(
        Kmn=Kmn, Knm=Knm, Lm=Lm, Knn=Knn, f=f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )


# q_sqrt is not care atm
def modified_conditional_with_lm(
    Kmn: tf.Tensor,
    Knm: tf.Tensor,
    Lm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov=False,
    q_sqrt: Optional[tf.Tensor] = None,
    white=False,
):
    r"""
    Has the same functionality as the `base_conditional` function, except that instead of
    `Kmm` this function accepts `Lm`, which is the Cholesky decomposition of `Kmm`.

    This allows `Lm` to be precomputed, which can improve performance.


    The modification is mainly based on calculating: Kmn * K^-1 * Knm, where Kmn and Knm are not exactly equal to each
    other's transpose, so we define an auxilary variable B to help computing

    Kmn * K^-1 * Knm = Knm'^T * L^T^-1 * L^-1 * Knm = (L^-1 knm')^T * (L^-1 * knm) = B^T * A
    :param q_sqrt
    """
    # compute kernel stuff
    num_func = tf.shape(f)[-1]  # R, denote multi-output number
    N = tf.shape(Kmn)[-1]
    M = tf.shape(f)[-2]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),
        (Knm, [..., "N", "M"]),
        (Lm, ["M", "M"]),
        (Knn, [..., "N", "N"] if full_cov else [..., "N"]),
        (f, ["M", "R"]),
    ]
    if q_sqrt is not None:
        shape_constraints.append(
            (q_sqrt, (["M", "R"] if q_sqrt.shape.ndims == 2 else ["R", "M", "M"]))
        )
    tf.debugging.assert_shapes(
        shape_constraints,
        message="base_conditional() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    leading_dims = tf.shape(Kmn)[:-2]

    # Compute the projection matrix A
    # Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [..., MC_num, M, M]
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]

    B = tf.linalg.triangular_solve(Lm, tf.linalg.matrix_transpose(Knm), lower=True)
    # compute the covariance due to the conditioning

    if full_cov:
        fvar = Knn - tf.linalg.matmul(B, A, transpose_a=True)  # [..., MC_num, N, N]
        cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -3), cov_shape)  # [..., R, N, N]
    else:
        fvar = Knn - tf.reduce_sum(B * A, -2)  # [..., N]
        cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -2), cov_shape)  # [..., R, N]

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)

    # construct the conditional mean
    f_shape = tf.concat([leading_dims, [M, num_func]], 0)  # [..., M, R]
    f = tf.broadcast_to(f, f_shape)  # [..., M, R]
    fmean = tf.linalg.matmul(A, f, transpose_a=True)  # [..., N, R]

    if q_sqrt is not None:
        raise NotImplementedError("Modified Condition has not done here yet")
        q_sqrt_dims = q_sqrt.shape.ndims
        if q_sqrt_dims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N]
        elif q_sqrt_dims == 3:
            L = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [R, M, M]
            L_shape = tf.shape(L)
            L = tf.broadcast_to(L, tf.concat([leading_dims, L_shape], 0))

            shape = tf.concat([leading_dims, [num_func, M, N]], axis=0)
            A_tiled = tf.broadcast_to(tf.expand_dims(A, -3), shape)
            LTA = tf.linalg.matmul(L, A_tiled, transpose_a=True)  # [R, M, N]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.shape.ndims))

        if full_cov:
            fvar = fvar + tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # [R, N]

    if not full_cov:
        fvar = tf.linalg.adjoint(fvar)  # [N, R]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),  # tensor included again for N dimension
        (f, [..., "M", "R"]),  # tensor included again for R dimension
        (fmean, [..., "N", "R"]),
        (fvar, [..., "R", "N", "N"] if full_cov else [..., "N", "R"]),
    ]
    tf.debugging.assert_shapes(shape_constraints, message="base_conditional() return values")

    return fmean, fvar


def modified_sample_mvn(mean, cov, full_cov, num_samples=None):
    """
    Returns a sample from a D-dimensional Multivariate Normal distribution
    :param mean: [..., N, D]
    :param cov: [..., N, D] or [..., N, D, D]
    :param full_cov: if `True` return a "full" covariance matrix, otherwise a "diag":
    - "full": cov holds the full covariance matrix (without jitter)
    - "diag": cov holds the diagonal elements of the covariance matrix
    :return: sample from the MVN of shape [..., (S), N, D], S = num_samples
    """
    shape_constraints = [
        (mean, [..., "N", "D"]),
        (cov, [..., "N", "D", "D"] if full_cov else [..., "N", "D"]),
    ]
    tf.debugging.assert_shapes(shape_constraints, message="sample_mvn() arguments")

    mean_shape = tf.shape(mean)
    S = num_samples if num_samples is not None else 1
    D = mean_shape[-1]
    leading_dims = mean_shape[:-2]

    if not full_cov:
        # mean: [..., N, D] and cov [..., N, D]
        eps_shape = tf.concat([leading_dims, [S], mean_shape[-2:]], 0)
        eps = tf.random.normal(eps_shape, dtype=default_float())  # [..., S, N, D]
        samples = mean[..., None, :, :] + tf.sqrt(cov)[..., None, :, :] * eps  # [..., S, N, D]

    else:
        # mean: [..., N, D] and cov [..., N, D, D]
        jittermat = (
            tf.eye(D, batch_shape=mean_shape[:-1], dtype=default_float()) * default_jitter()
        )  # [..., N, D, D]
        eps_shape = tf.concat([mean_shape, [S]], 0)
        eps = tf.random.normal(eps_shape, dtype=default_float())  # [..., N, D, S]
        chol = tf.linalg.cholesky(cov + jittermat)  # [..., N, D, D]
        samples = mean[..., None] + tf.linalg.matmul(chol, eps)  # [..., N, D, S]
        samples = leading_transpose(samples, [..., -1, -3, -2])  # [..., S, N, D]

    shape_constraints = [
        (mean, [..., "N", "D"]),
        (samples, [..., "S", "N", "D"]),
    ]
    tf.debugging.assert_shapes(shape_constraints, message="sample_mvn() return values")

    if num_samples is None:
        return tf.squeeze(samples, axis=-3)  # [..., N, D]
    return samples  # [..., S, N, D]
