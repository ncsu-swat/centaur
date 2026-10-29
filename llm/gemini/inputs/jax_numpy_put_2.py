
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def put_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, basic indices
    a = np.zeros(5, dtype=np.int32)
    ind = np.array([0, 2, 4], dtype=np.int32)
    v = np.array([10, 20, 30], dtype=np.int32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 2: 1D float array, wrap mode
    a = np.ones(6, dtype=np.float32)
    ind = np.array([1, 3, 5], dtype=np.int32)
    v = np.array([-1.5, 2.5, -3.5], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "wrap", "inplace": False}
    )

    # Input 3: 2D float64 array, clip mode
    a = np.zeros((3, 4), dtype=np.float64)
    ind = np.array([0, 5, 11], dtype=np.int64)
    v = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 4: 2D int64 array, wrap mode
    a = np.arange(12, dtype=np.int64).reshape((4, 3))
    ind = np.array([2, 6, 10], dtype=np.int32)
    v = np.array([99, 100, 101], dtype=np.int64)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "wrap", "inplace": False}
    )

    # Input 5: 3D float32 array
    a = np.zeros((2, 3, 4), dtype=np.float32)
    ind = np.array([0, 7, 15, 23], dtype=np.int32)
    v = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 6: 1D boolean array
    a = np.zeros(8, dtype=bool)
    ind = np.array([0, 3, 6], dtype=np.int32)
    v = np.array([True, True, True], dtype=bool)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "wrap", "inplace": False}
    )

    # Input 7: 2D float32 array
    a = np.zeros((4, 4), dtype=np.float32)
    ind = np.array([1, 5, 9, 13], dtype=np.int32)
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 8: Single index replacement
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    ind = np.array([2], dtype=np.int32)
    v = np.array([999.0], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "wrap", "inplace": False}
    )

    # Input 9: 4D int32 array
    a = np.zeros((2, 2, 2, 2), dtype=np.int32)
    ind = np.array([0, 1, 3, 5], dtype=np.int32)
    v = np.array([10, 20, 30, 40], dtype=np.int32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 10: 1D int32 array with multiple updates
    a = np.ones(10, dtype=np.int32)
    ind = np.array([0, 4, 8], dtype=np.int32)
    v = np.array([100, 200, 300], dtype=np.int32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 11: 2D int32 array with negative values
    a = np.zeros((3, 3), dtype=np.int32)
    ind = np.array([1, 4, 7], dtype=np.int32)
    v = np.array([-10, -20, -30], dtype=np.int32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "wrap", "inplace": False}
    )

    return list_of_inputs


generated_inputs["jax.numpy.put_2"] = put_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_2'.")


check_valid('jax.numpy.put', generated_inputs['jax.numpy.put_2'], lib="jax", suffix=2)
