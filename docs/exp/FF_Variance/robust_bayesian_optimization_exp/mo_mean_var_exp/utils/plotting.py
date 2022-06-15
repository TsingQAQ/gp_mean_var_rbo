import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from trieste.acquisition.multi_objective.partition import non_dominated


def format_point_markers(
    num_pts,
    num_init=None,
    idx_best=None,
    mask_fail=None,
    m_init="x",
    m_add="o",
    c_pass="tab:green",
    c_fail="tab:red",
    c_best="tab:purple",
):
    """
    Prepares point marker styles according to some BO factors
    :param num_pts: total number of BO points
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
    :param m_init: marker for the initial BO points
    :param m_add: marker for the other BO points
    :param c_pass: color for the regular BO points
    :param c_fail: color for the failed BO points
    :param c_best: color for the best BO points
    :return: 2 string vectors col_pts, mark_pts containing marker styles and colors
    """
    if num_init is None:
        num_init = num_pts
    col_pts = np.repeat(c_pass, num_pts)
    col_pts = col_pts.astype("<U15")
    mark_pts = np.repeat(m_init, num_pts)
    mark_pts[num_init:] = m_add
    if mask_fail is not None:
        col_pts[np.where(mask_fail)] = c_fail
    if idx_best is not None:
        col_pts[idx_best] = c_best

    return col_pts, mark_pts


def plot_mobo_points_in_obj_space(
    obs_values,
    num_init=None,
    mask_fail=None,
    figsize=None,
    xlabel="Obj 1",
    ylabel="Obj 2",
    zlabel="Obj 3",
    title=None,
    m_init="x",
    m_add="o",
    c_pass="tab:green",
    c_fail="tab:red",
    c_pareto="tab:purple",
    only_plot_pareto=False,
):
    """
    Adds scatter points in objective space, used for multi-objective optimization (2 or 3 objectives only).
    Markers and colors are chosen according to BO factors.

    :param obs_values: TF Tensor or numpy array of objective values, shape (N, 2) or (N, 3).
    :param num_init: initial number of BO points
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
    :param figsize: Size of the figure.
    :param xlabel: Label of the X axis.
    :param ylabel: Label of the Y axis.
    :param zlabel: Label of the Z axis (in 3d case).
    :param title: Title of the plot.
    :param m_init: Marker for initial points.
    :param m_add: Marker for the points observed during the BO loop.
    :param c_pass: color for the regular BO points
    :param c_fail: color for the failed BO points
    :param c_pareto: color for the Pareto front points
    :param only_plot_pareto: if set to `True`, only plot the pareto points. Default is `False`.
    """
    obj_num = obs_values.shape[-1]
    tf.debugging.assert_shapes([])
    assert obj_num == 2 or obj_num == 3, NotImplementedError(
        f"Only support 2/3-objective functions but found: {obj_num}"
    )

    _, dom = non_dominated(obs_values)
    idx_pareto = (
        np.where(dom == 0) if mask_fail is None else np.where(np.logical_and(dom == 0, ~mask_fail))
    )

    pts = obs_values.numpy() if tf.is_tensor(obs_values) else obs_values
    num_pts = pts.shape[0]

    col_pts, mark_pts = format_point_markers(
        num_pts, num_init, idx_pareto, mask_fail, m_init, m_add, c_pass, c_fail, c_pareto
    )
    if only_plot_pareto:
        col_pts = col_pts[idx_pareto]
        mark_pts = mark_pts[idx_pareto]
        pts = pts[idx_pareto]

    if obj_num == 2:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for i in range(pts.shape[0]):
        ax.scatter(*pts[i], c=col_pts[i], marker=mark_pts[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if obj_num == 3:
        ax.set_zlabel(zlabel)
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_mobo_history(
    obs_values,
    metric_func,
    num_init=None,
    mask_fail=None,
    figsize=None,
):
    """
    Draw the performance measure for multi-objective optimization
    :param obs_values:
    :param metric_func: a callable function calculate metric score
                        metric = measure_func(observations)
    :param num_init:
    :param mask_fail:
    :param figsize
    """

    fig, ax = plt.subplots(figsize=figsize)
    size, obj_num = obs_values.shape

    if mask_fail is not None:
        obs_values[mask_fail] = [np.inf] * obj_num

    _idxs = np.arange(1, size + 1)
    ax.plot(_idxs, [metric_func(obs_values[:pts, :]) for pts in _idxs], color="tab:orange")
    ax.axvline(x=num_init - 0.5, color="tab:blue")
    return fig, ax
