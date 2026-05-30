
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D array, int32
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {
        "a": a,
        "dtype": "int32",
        "shape": [5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32 with different shape
    a = np.random.randn(2, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "dtype": "float32",
        "shape": [3, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, float64
    a = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "dtype": "float64",
        "shape": [4, 4, 4]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array, bool
    a = np.array([[[[True, False]]]], dtype=np.bool_)
    input_dict = {
        "a": a,
        "dtype": "bool",
        "shape": [2, 2, 1, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex array
    a = np.array([1+2j, 3+4j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "dtype": "complex64",
        "shape": [3, 3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D array (scalar-like)
    a = np.array(42.0, dtype=np.float32)
    input_dict = {
        "a": a,
        "dtype": "float32",
        "shape": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int16 array
    a = np.arange(10, dtype=np.int16)
    input_dict = {
        "a": a,
        "dtype": "int16",
        "shape": [5, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with negative values, float32 to float64
    a = np.array([-1.5, -2.5, -3.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "dtype": "float64",
        "shape": [3, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8 array
    a = np.random.randint(0, 256, size=(4, 4), dtype=np.uint8)
    input_dict = {
        "a": a,
        "dtype": "uint8",
        "shape": [8, 8]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64 array
    a = np.array([100000000000], dtype=np.int64)
    input_dict = {
        "a": a,
        "dtype": "int64",
        "shape": [1, 5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.zeros_like_6"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_6'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_6'], lib="jax", suffix=6)
