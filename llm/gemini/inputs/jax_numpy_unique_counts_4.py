
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_unique_counts_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers
    x = np.array([1, 2, 2, 3, 3, 3], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 3,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of floats with negative values
    x = np.array([-1.5, 2.3, -1.5, 0.0, 4.2], dtype=np.float32)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of int64 with size larger than unique count (padding expected)
    x = np.array([[1, 2], [2, 3]], dtype=np.int64)
    input_dict = {
        "x": x,
        "size": 5,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array
    x = np.random.randn(2, 3, 2).astype(np.float64)
    input_dict = {
        "x": x,
        "size": 10,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D boolean array
    x = np.array([True, False, True, False, False], dtype=np.bool_)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of uniform values
    x = np.ones(10, dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 1,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large 1D array with random integers
    x = np.random.randint(-10, 10, size=(50,)).astype(np.int32)
    input_dict = {
        "x": x,
        "size": 15,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D int32 array
    x = np.random.randint(0, 5, size=(2, 2, 2, 2)).astype(np.int32)
    input_dict = {
        "x": x,
        "size": 4,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element 1D array with padding size
    x = np.array([42], dtype=np.int64)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array with duplicate patterns
    x = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float32)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_counts_4"] = jax_numpy_unique_counts_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_counts_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_counts_4'.")


check_valid('jax.numpy.unique_counts', generated_inputs['jax.numpy.unique_counts_4'], lib="jax", suffix=4)
