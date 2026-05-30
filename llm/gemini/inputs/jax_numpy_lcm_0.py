
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lcm_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays with int32
    x1 = np.array([12, 18, 24], dtype=np.int32)
    x2 = np.array([5, 10, 15], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays with negative values, int32
    x1 = np.array([[-12, 15], [8, -6]], dtype=np.int32)
    x2 = np.array([[18, -10], [-4, 9]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays with int64
    x1 = np.array([100000000, 200000000], dtype=np.int64)
    x2 = np.array([300000000, 400000000], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting case: 2D array (1, 3) and 2D array (3, 1) with int32
    x1 = np.array([[12, 18, 24]], dtype=np.int32)
    x2 = np.array([[5], [10], [15]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays with int32
    x1 = np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2)
    x2 = np.arange(9, 17, dtype=np.int32).reshape(2, 2, 2)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D arrays with int64
    x1 = np.ones((2, 2, 2, 2), dtype=np.int64) * 6
    x2 = np.ones((2, 2, 2, 2), dtype=np.int64) * 8
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Inputs containing zeros (int32)
    x1 = np.array([0, 5, 0], dtype=np.int32)
    x2 = np.array([3, 0, 0], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting 1D array of size 1 and 2D array (int32)
    x1 = np.array([15], dtype=np.int32)
    x2 = np.array([[5, 10, 25], [30, 45, 60]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 array with negative values
    x1 = np.array([-2, -3, -4, -5], dtype=np.int64)
    x2 = np.array([3, 4, 5, 6], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with mixed positive and negative values, int32
    x1 = np.array([12, -15, 20, -35], dtype=np.int32)
    x2 = np.array([-18, 25, -30, 42], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
