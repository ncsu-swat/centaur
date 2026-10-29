
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def put_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, mode 'clip'
    a = np.zeros(5, dtype=np.int32)
    ind = np.array([0, 2, 4], dtype=np.int32)
    v = np.array([10, 20, 30], dtype=np.int32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 2: 1D integer array with out-of-bounds index, mode 'wrap'
    a = np.zeros(5, dtype=np.int32)
    ind = np.array([0, 2, 6], dtype=np.int32)
    v = np.array([10, 20, 30], dtype=np.int32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "wrap", "inplace": False}
    )

    # Input 3: 2D float32 array, mode 'clip'
    a = np.zeros((3, 4), dtype=np.float32)
    ind = np.array([1, 5, 11], dtype=np.int64)
    v = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 4: 2D float64 array with out-of-bounds indices, mode 'wrap'
    a = np.ones((2, 3), dtype=np.float64)
    ind = np.array([6, 7, 8], dtype=np.int32)
    v = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "wrap", "inplace": False}
    )

    # Input 5: 3D int64 array, mode 'clip'
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    ind = np.array([0, 10, 23], dtype=np.int32)
    v = np.array([99, 100, 101], dtype=np.int64)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 6: 1D bool array, mode 'clip'
    a = np.array([True, False, False, True], dtype=bool)
    ind = np.array([1, 2], dtype=np.int32)
    v = np.array([True, True], dtype=bool)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 7: 1D complex64 array, mode 'wrap'
    a = np.zeros(4, dtype=np.complex64)
    ind = np.array([0, 3, 5], dtype=np.int32)
    v = np.array([1.0 + 2.0j, 3.0 - 1.0j, -2.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "wrap", "inplace": False}
    )

    # Input 8: 4D float32 array, mode 'clip'
    a = np.zeros((2, 2, 2, 2), dtype=np.float32)
    ind = np.array([0, 7, 15], dtype=np.int32)
    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 9: 1D float32 array with negative indices, mode 'wrap'
    a = np.zeros(6, dtype=np.float32)
    ind = np.array([-1, -3, -5], dtype=np.int32)
    v = np.array([5.5, 4.4, 3.3], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "wrap", "inplace": False}
    )

    # Input 10: 2D uint8 array, single index and value, mode 'clip'
    a = np.full((3, 3), 255, dtype=np.uint8)
    ind = np.array([4], dtype=np.int32)
    v = np.array([0], dtype=np.uint8)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    # Input 11: 1D int16 array with repeated indices, mode 'clip'
    a = np.zeros(5, dtype=np.int16)
    ind = np.array([1, 1, 3], dtype=np.int32)
    v = np.array([10, 20, 30], dtype=np.int16)
    list_of_inputs.append(
        {"a": a, "ind": ind, "v": v, "mode": "clip", "inplace": False}
    )

    return list_of_inputs


generated_inputs["jax.numpy.put_4"] = put_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_4'.")


check_valid('jax.numpy.put', generated_inputs['jax.numpy.put_4'], lib="jax", suffix=4)
