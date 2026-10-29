
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ediff1d_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    ary = np.array([2, 3, 5, 9, 1, 4], dtype=np.int32)
    to_end = 10
    to_begin = -5
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 2: 1D float32 array
    ary = np.array([1.5, -2.3, 4.1, 0.0, 7.8], dtype=np.float32)
    to_end = 0
    to_begin = 3
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 3: 2D int64 array
    ary = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    to_end = -1
    to_begin = 100
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 4: 2D float64 array with negative and positive values
    ary = np.array([[-1.0, 2.5], [3.0, -4.5]], dtype=np.float64)
    to_end = 2
    to_begin = -2
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 5: 3D int32 array
    ary = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    to_end = 99
    to_begin = 0
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 6: Single-element 1D array
    ary = np.array([42], dtype=np.int32)
    to_end = 5
    to_begin = -10
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 7: 1D int32 array with negative values
    ary = np.array([1000, -2000, 3000, -4000], dtype=np.int32)
    to_end = 50
    to_begin = -50
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 8: 4D float32 array
    ary = np.ones((2, 2, 2, 2), dtype=np.float32)
    to_end = -7
    to_begin = 7
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 9: 1D int64 array
    ary = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    to_end = 1
    to_begin = 2
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 10: 2D int32 array with shape (1, 8)
    ary = np.array([[7, -3, 2, 11, -9, 0, 5, 8]], dtype=np.int32)
    to_end = -12
    to_begin = 25
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 11: 3D float32 array
    ary = np.random.randn(3, 2, 4).astype(np.float32)
    to_end = 0
    to_begin = 0
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    return list_of_inputs


generated_inputs["jax.numpy.ediff1d_2"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_2'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_2'], lib="jax", suffix=2)
