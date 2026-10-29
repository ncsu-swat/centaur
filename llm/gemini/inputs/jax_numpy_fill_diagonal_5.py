
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import jax.numpy as jnp
import numpy as np


class ArrayTuple(tuple, jax.Array):
    __abstractmethods__ = frozenset()

    def __jax_array__(self):
        return jnp.asarray(tuple(self))

    def __array__(self, dtype=None):
        return np.asarray(tuple(self), dtype=dtype)

    def __deepcopy__(self, memo):
        return ArrayTuple(copy.deepcopy(tuple(self), memo))

    def min(self, *args, **kwargs):
        return min(self)

    def max(self, *args, **kwargs):
        return max(self)


def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix with single-element tuple
    a = np.zeros((3, 3), dtype=np.float32)
    val = ArrayTuple((1.5,))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 2: 2D square matrix with exact matching tuple length
    a = np.zeros((4, 4), dtype=np.int32)
    val = ArrayTuple((1, 2, 3, 4))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 3: 2D rectangular matrix (wide) with repeating tuple
    a = np.ones((3, 5), dtype=np.float64)
    val = ArrayTuple((-1.0, -2.0))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 4: 2D rectangular matrix (tall) with truncated tuple
    a = np.zeros((6, 3), dtype=np.int64)
    val = ArrayTuple((10, 20, 30, 40, 50))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 5: 3D cube array with scalar tuple
    a = np.zeros((3, 3, 3), dtype=np.float32)
    val = ArrayTuple((7.0,))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 6: 4D hypercube array
    a = np.zeros((2, 2, 2, 2), dtype=np.float32)
    val = ArrayTuple((0.5, -0.5))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 7: Boolean array
    a = np.zeros((4, 4), dtype=bool)
    val = ArrayTuple((True, False))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 8: 2D square float64 array
    a = np.zeros((3, 3), dtype=np.float64)
    val = ArrayTuple((1.0, -3.0, 4.0))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 9: 2D square matrix with negative numbers
    a = np.full((5, 5), -9, dtype=np.int32)
    val = ArrayTuple((-1, -2, -3))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 10: 3D cube array with 4 elements per dim
    a = np.ones((4, 4, 4), dtype=np.int64)
    val = ArrayTuple((100, 200, 300, 400))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    # Input 11: 5D hypercube array with float values
    a = np.zeros((2, 2, 2, 2, 2), dtype=np.float64)
    val = ArrayTuple((3.14159,))
    list_of_inputs.append(
        {"a": a, "val": val, "wrap": False, "inplace": False}
    )

    return [copy.deepcopy(x) for x in list_of_inputs]


generated_inputs["jax.numpy.fill_diagonal_5"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_5'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_5'], lib="jax", suffix=5)
