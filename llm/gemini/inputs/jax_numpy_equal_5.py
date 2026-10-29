
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def equal_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    input_dict = {
        "x": 0,
        "y": np.array([0, 1, 2, 0, -1], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D integer array with negative integer
    input_dict = {
        "x": -5,
        "y": np.array([[-5, 2], [3, -5]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float array with positive integer
    input_dict = {
        "x": 1,
        "y": np.array([1.0, 2.5, 1.0, 0.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float array
    input_dict = {
        "x": 42,
        "y": np.zeros((2, 3, 4), dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D tensor (scalar array)
    input_dict = {
        "x": 10,
        "y": np.array(10, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensor with uint8 dtype
    input_dict = {
        "x": 255,
        "y": np.full((1, 2, 2, 3), 255, dtype=np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D boolean tensor
    input_dict = {
        "x": 1,
        "y": np.array([[True, False], [False, True]], dtype=np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty array (0 elements)
    input_dict = {
        "x": 0,
        "y": np.array([], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer with int64 array
    input_dict = {
        "x": 1000000,
        "y": np.array([1000000, -1000000, 0], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float16 array with negative value
    input_dict = {
        "x": -1,
        "y": np.array([[-1.0, 0.0], [1.0, -1.0]], dtype=np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Complex numbers array
    input_dict = {
        "x": 2,
        "y": np.array([2.0 + 0.0j, 2.0 + 1.0j, -2.0 + 0.0j], dtype=np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.equal_5"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_5'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_5'], lib="jax", suffix=5)
