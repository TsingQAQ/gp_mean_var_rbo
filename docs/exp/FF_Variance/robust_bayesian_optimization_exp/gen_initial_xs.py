"""
Generate initial DOE for benchmarking
This is to assure that the comparison of different acq starts from exactly the same data
"""
import json
import os

import numpy as np

from trieste.objectives import single_objectives
from trieste.space import Box


def gen_doe():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--number_of_initial_xs", type=int)
    parser.add_argument("-b", "--batch_size_of_doe", type=int)
    parser.add_argument("-pb", "--problem", type=str)
    parser.add_argument("-kw_pb", type=json.loads, default={})
    parser.add_argument("-tau", type=float, default=None)

    _args = parser.parse_args()
    doe_num = _args.number_of_initial_xs
    doe_repeat = _args.batch_size_of_doe
    pb_name = _args.problem
    pb_kw = _args.kw_pb
    tau = _args.tau

    pb = getattr(single_objectives, pb_name)(**pb_kw)
    for i in range(doe_repeat):
        if tau is None:
            xs = Box(*pb.bounds).sample(doe_num).numpy()
        else:
            xs = Box(*pb.bounds).discretize_v2(tau).sample(doe_num).numpy()
        _mo_path = os.path.join("mo_mean_var_exp", "cfg", "initial_xs")
        _smv_path = os.path.join("scalar_mean_var_exp", "cfg", "initial_xs")
        _vs_path = os.path.join("var_as_con_acq_exp", "cfg", "initial_xs")
        os.makedirs(_mo_path, exist_ok=True)
        os.makedirs(_smv_path, exist_ok=True)
        os.makedirs(_vs_path, exist_ok=True)
        np.savetxt(os.path.join(_mo_path, pb_name, f"xs_{i}.txt"), xs)
        np.savetxt(os.path.join(_smv_path, pb_name, f"xs_{i}.txt"), xs)
        np.savetxt(os.path.join(_vs_path, pb_name, f"xs_{i}.txt"), xs)


if __name__ == "__main__":
    gen_doe()
