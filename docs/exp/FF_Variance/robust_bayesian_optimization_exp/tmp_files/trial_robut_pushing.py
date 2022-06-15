import tensorflow as tf
from PyOptimize.utils.visualization import view_2D_function_in_contour

from trieste.objectives.extra_problems.robot_pushing.problem_wrapper import (
    Robot_Pushing_2D,
    Robot_Pushing_3D,
)

if __name__ == "__main__":
    pb = Robot_Pushing_2D(10, 10, 1).objective()

    view_2D_function_in_contour(pb, [[0, 1], [0, 1]], plot_fidelity=20, show=True, colorbar=True)
    # print(pb(tf.constant([[1.0, 1.0, 1], [0, 0, 0]])))

    from scipy.optimize import Bounds, minimize

    # opt_obj = lambda at: tf.squeeze(pb(at))
    # res = minimize(opt_obj, tf.constant([[0.2, 0.2, 0.3]], dtype=tf.float64),
    #                bounds=Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    # print(res)
