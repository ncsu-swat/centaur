
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vander_inputs():
    list_of_inputs = []

    # Input 1: Simple integers, decreasing powers
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {
        "x": x,
        "N": 4,
        "increasing": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floating point values, increasing powers
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "N": 3,
        "increasing": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative and zero values, int64 dtype, N smaller than length of x
    x = np.array([-1, 0, 1], dtype=np.int64)
    input_dict = {
        "x": x,
        "N": 2,
        "increasing": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 dtype, N larger than length of x, increasing powers
    x = np.array([0.5, 1.5, -2.5, 3.5], dtype=np.float64)
    input_dict = {
        "x": x,
        "N": 5,
        "increasing": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element array, N = 1
    x = np.array([10], dtype=np.int32)
    input_dict = {
        "x": x,
        "N": 1,
        "increasing": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Alternating signs, larger N, increasing powers
    x = np.array([2, -2, 3, -3, 4, -4], dtype=np.int32)
    input_dict = {
        "x": x,
        "N": 6,
        "increasing": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float32 values with decimal parts, decreasing powers
    x = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "N": 3,
        "increasing": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative integers only, increasing powers
    x = np.array([-5, -4, -3, -2, -1], dtype=np.int64)
    input_dict = {
        "x": x,
        "N": 4,
        "increasing": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All zero inputs, decreasing powers
    x = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {
        "x": x,
        "N": 3,
        "increasing": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger range array, float64 dtype, N=2 (linear/constant terms)
    x = np.arange(1, 11, dtype=np.float64)
    input_dict = {
        "x": x,
        "N": 2,
        "increasing": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.vander"] = vander_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vander' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vander'.")


check_valid('jax.numpy.vander', generated_inputs['jax.numpy.vander'], lib="jax", suffix=0)
