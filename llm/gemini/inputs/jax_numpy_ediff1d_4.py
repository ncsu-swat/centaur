
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ediff1d_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with 1D arrays for to_begin and to_end
    ary = np.array([1, 2, 4, 7, 0], dtype=np.int32)
    to_end = np.array([99, 100], dtype=np.int32)
    to_begin = np.array([-5], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 2: 1D float array with float32 arrays
    ary = np.array([1.5, -2.3, 3.7, 4.1], dtype=np.float32)
    to_end = np.array([0.0, -1.0], dtype=np.float32)
    to_begin = np.array([10.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 3: 2D integer array with 0D (scalar) arrays
    ary = np.array([[1, 2], [3, 4]], dtype=np.int32)
    to_end = np.array(5, dtype=np.int32)
    to_begin = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 4: 1D negative float array with float64 arrays
    ary = np.array([-10.0, -20.0, -30.0], dtype=np.float64)
    to_end = np.array([-40.0, -50.0], dtype=np.float64)
    to_begin = np.array([-5.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 5: 3D nested array with multi-dimensional to_end array
    ary = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    to_end = np.array([[9, 10], [11, 12]], dtype=np.int32)
    to_begin = np.array([0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 6: 1D int16 array with int16 arrays
    ary = np.array([10, -20, 30, -40, 50], dtype=np.int16)
    to_end = np.array([100], dtype=np.int16)
    to_begin = np.array([0], dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 7: Single-element array
    ary = np.array([42], dtype=np.int64)
    to_end = np.array([1, 2], dtype=np.int64)
    to_begin = np.array([3, 4], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 8: 1D array with widely varying magnitudes
    ary = np.array([1e-5, 1e5, -1e-5, -1e5], dtype=np.float32)
    to_end = np.array([0.0], dtype=np.float32)
    to_begin = np.array([1e2, -1e2], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 9: 2D array of floats with 2D to_begin array
    ary = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    to_end = np.array([1.0], dtype=np.float32)
    to_begin = np.array([[0.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 10: 1D array of large 64-bit integers
    ary = np.array([10000000000, 20000000000, 15000000000], dtype=np.int64)
    to_end = np.array([30000000000], dtype=np.int64)
    to_begin = np.array([0], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    # Input 11: Complex numbers array
    ary = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], dtype=np.complex64)
    to_end = np.array([7.0 + 8.0j], dtype=np.complex64)
    to_begin = np.array([0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "to_end": to_end, "to_begin": to_begin})
    )

    return list_of_inputs


generated_inputs["jax.numpy.ediff1d_4"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_4'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_4'], lib="jax", suffix=4)
