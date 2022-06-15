from functools import partial
from typing import Optional, Union

import tensorflow as tf


def sequential_batch(max_batch_element_num: Union[int, bool, None] = None):
    def sequential_input_wrapper(func):
        """
        A sequential input decomposer to decompose input: x to decompose_batch_size,
        this helps avoiding OOM issue in case x is decomposable

        we assume x is shape [N, dim], so we concatenate -2 axis
        """

        def wrapper(*args, **kw):
            x = args[0]
            if (
                x.shape[0] == 1
                or max_batch_element_num is None
                or x.shape[0] <= max_batch_element_num
            ):
                return func(*args, **kw)
            else:
                # if x.shape[0] != 1 and x.shape[0] > max_batch_element_num:
                if max_batch_element_num == 1:  # maximum 1, fully sequential
                    splited_data = tf.split(x, x.shape[0])
                else:  # use customed max_batch_element_num
                    # as tf.split must deal with a max_batch_element_num, we need some bit extra hack here:
                    round_split_num = tf.cast(
                        tf.math.floor(x.shape[0] / max_batch_element_num), dtype=tf.int32
                    ).numpy()
                    rest_x_size = tf.cast(
                        tf.math.floormod(x.shape[0], round_split_num), dtype=tf.int32
                    ).numpy()
                    if rest_x_size != 0:
                        splited_data = tf.split(x[:-rest_x_size], round_split_num)
                        splited_data.append(x[-rest_x_size:])
                    else:
                        splited_data = tf.split(x, round_split_num)
                # splited_data = tf.concat([data[None] for data in splited_data], axis=0)
                # splited_data = [data[None] for data in splited_data]
                res = []
                if len(args) > 1:
                    [
                        res.append(partial(func, **kw)(split_data, *args[1:]))
                        for split_data in splited_data
                    ]
                else:
                    [res.append(partial(func, **kw)(split_data)) for split_data in splited_data]
                target_func_values = tf.concat(
                    res, -2
                )  # we assume x is shape [N, dim], so we concatenate -2 axis
                return target_func_values

        return wrapper

    return sequential_input_wrapper


if __name__ == "__main__":

    @sequential_batch(max_batch_element_num=100)
    def hellow_decorator(x):
        return tf.reduce_sum(x, -1, keepdims=True)

    input = tf.random.normal(shape=(107, 2))
    print(hellow_decorator(input))
