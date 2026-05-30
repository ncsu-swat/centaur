
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iterable_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers
    y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 2: 0D array (scalar array) of float
    y = np.array(5.5, dtype=np.float32)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 3: 2D array of floats
    y = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 4: 3D array of booleans
    y = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 5: 1D array of negative integers
    y = np.array([-10, -20, -30], dtype=np.int64)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 6: 1D array of complex numbers
    y = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 7: Empty 1D array
    y = np.array([], dtype=np.float64)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 8: 4D array of uint8
    y = np.ones((1, 2, 2, 1), dtype=np.uint8)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 9: 1D array of inf and nan
    y = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 10: 2D array of float16
    y = np.zeros((3, 3), dtype=np.float16)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    # Input 11: 1D array of int16
    y = np.arange(10, dtype=np.int16)
    list_of_inputs.append({"y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.iterable_1"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_1'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_1'], lib="jax", suffix=1)
