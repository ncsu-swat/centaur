
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lcm_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays of same shape
    x1 = np.array([12, 18, 24], dtype=np.int32)
    x2 = np.array([5, 10, 15], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 0D scalar arrays
    x1 = np.array(12, dtype=np.int32)
    x2 = np.array(18, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D arrays with negative values
    x1 = np.array([[-12, 15], [20, -30]], dtype=np.int32)
    x2 = np.array([[18, -25], [-16, 45]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting (3, 1) with (1, 4)
    x1 = np.array([[6], [8], [10]], dtype=np.int32)
    x2 = np.array([[2, 3, 4, 5]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 1D array broadcast with 0D array
    x1 = np.array([6, 9, 12, 15], dtype=np.int64)
    x2 = np.array(12, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 3D arrays with zeros
    x1 = np.array([[[0, 4], [6, 8]], [[10, 0], [14, 16]]], dtype=np.int32)
    x2 = np.array([[[5, 0], [9, 12]], [[0, 15], [21, 24]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 1D int64 arrays
    x1 = np.array([14, 21, 35], dtype=np.int64)
    x2 = np.array([21, 28, 42], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: int32 arrays with negative and positive values
    x1 = np.array([-3, -7, 11, -13], dtype=np.int32)
    x2 = np.array([4, 6, -8, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Large int64 values
    x1 = np.array([1000003, 1000033], dtype=np.int64)
    x2 = np.array([1000037, 1000039], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 4D arrays with broadcasting
    x1 = np.random.randint(1, 20, size=(2, 1, 3, 1), dtype=np.int32)
    x2 = np.random.randint(1, 20, size=(1, 2, 1, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 2D int64 broadcasting
    x1 = np.array([[12, 24, 36]], dtype=np.int64)
    x2 = np.array([[2], [3], [4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.lcm"] = lcm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.lcm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.lcm'.")


check_valid('jax.numpy.lcm', generated_inputs['jax.numpy.lcm'], lib="jax", suffix=0)
