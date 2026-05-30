
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def shape_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: 2D float32 array
    a = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: 3D float64 array
    a = np.random.randn(2, 2, 2).astype(np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: 4D int64 array
    a = np.arange(24).reshape(2, 3, 2, 2).astype(np.int64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: 5D uint8 array
    a = np.ones((1, 2, 1, 3, 1), dtype=np.uint8)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: 0D array (scalar)
    a = np.array(42.0, dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: 1D boolean array
    a = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: 2D complex array
    a = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: 3D array with negative values
    a = np.array([[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]], dtype=np.int16)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: High-dimensional empty-like array (10D)
    a = np.empty((1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    return list_of_inputs

generated_inputs["jax.numpy.shape_1"] = shape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.shape_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.shape_1'.")


check_valid('jax.numpy.shape', generated_inputs['jax.numpy.shape_1'], lib="jax", suffix=1)
