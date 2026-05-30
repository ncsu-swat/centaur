
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def power_inputs():
    list_of_inputs = []

    # Input 1: Scalar int32
    input_dict = {
        "x1": np.int32(5),
        "x2": np.int32(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar int64
    input_dict = {
        "x1": np.int64(-4),
        "x2": np.int64(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, same size (int32)
    input_dict = {
        "x1": np.array([1, 2, 3], dtype=np.int32),
        "x2": np.array([3, 2, 1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, same size (int64)
    input_dict = {
        "x1": np.array([[1, -2], [3, -4]], dtype=np.int64),
        "x2": np.array([[2, 3], [1, 0]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broad casting: 1D x1 and 2D x2
    input_dict = {
        "x1": np.array([2, 3, 4], dtype=np.int32),
        "x2": np.array([[1, 2, 3], [0, 1, 2]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broad casting: 2D x1 and 1D x2
    input_dict = {
        "x1": np.array([[2], [3]], dtype=np.int32),
        "x2": np.array([1, 2, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D x1 and scalar x2
    input_dict = {
        "x1": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "x2": np.int32(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar x1 and 3D x2
    input_dict = {
        "x1": np.int32(-2),
        "x2": np.array([[[1, 2], [3, 0]], [[2, 1], [0, 3]]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D arrays (broadcasting compatible)
    input_dict = {
        "x1": np.array([[[[1, 2]], [[3, 4]]]], dtype=np.int64),
        "x2": np.array([[[[2, 3], [1, 2]]]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D arrays with 0 and negative bases, positive exponents
    input_dict = {
        "x1": np.array([-3, 0, 5, -2], dtype=np.int32),
        "x2": np.array([3, 5, 2, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.power_6"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_6'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_6'], lib="jax", suffix=6)
