
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ediff1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int32 arrays
    ary = np.array([2, 3, 5, 9, 1, 4], dtype=np.int32)
    to_end = np.array([20, 30], dtype=np.int32)
    to_begin = np.array([-10], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 2: 1D float32 array with negative and fractional values
    ary = np.array([-1.5, 2.0, -3.5, 4.5, -5.0], dtype=np.float32)
    to_end = np.array([99.0], dtype=np.float32)
    to_begin = np.array([-99.0, 0.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 3: 2D array with 1D prepend/append arrays
    ary = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int64)
    to_end = np.array([10, 20, 30], dtype=np.int64)
    to_begin = np.array([0], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 4: 3D float64 array
    ary = np.random.randn(2, 3, 4).astype(np.float64)
    to_end = np.array([1.23, 4.56], dtype=np.float64)
    to_begin = np.array([-7.89], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 5: Single element 1D array
    ary = np.array([42], dtype=np.int32)
    to_end = np.array([100, 200], dtype=np.int32)
    to_begin = np.array([-50, -25], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 6: 0D scalar-like arrays
    ary = np.array([10.0, 20.0, 35.0, 55.0], dtype=np.float32)
    to_end = np.array(100.0, dtype=np.float32)
    to_begin = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 7: Empty 1D arrays for to_end and to_begin
    ary = np.array([5, 10, 15, 20, 25], dtype=np.int32)
    to_end = np.array([], dtype=np.int32)
    to_begin = np.array([], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 8: 1D float64 array with multiple values
    ary = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    to_end = np.array([5.5, 6.5], dtype=np.float64)
    to_begin = np.array([-0.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 9: 4D array with int32 data type
    ary = np.arange(16, dtype=np.int32).reshape(2, 2, 2, 2)
    to_end = np.array([50, 60], dtype=np.int32)
    to_begin = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 10: 2D arrays for to_end and to_begin
    ary = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    to_end = np.array([[25.0, 36.0], [49.0, 64.0]], dtype=np.float32)
    to_begin = np.array([[-1.0, 0.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    return list_of_inputs


generated_inputs["jax.numpy.ediff1d_1"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_1'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_1'], lib="jax", suffix=1)
