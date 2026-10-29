
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ones_inputs():
    list_of_inputs = []

    # Input 1: 1D list, float32
    input_dict = {"shape": [5], "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D list, int32
    input_dict = {"shape": [2, 3], "dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D list, bool
    input_dict = {"shape": [2, 2, 4], "dtype": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D list (scalar shape), float64
    input_dict = {"shape": [], "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D list, int64
    input_dict = {"shape": [1, 2, 3, 4], "dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D with zero dimension, float32
    input_dict = {"shape": [0, 5], "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D list with 1 element, complex64
    input_dict = {"shape": [1], "dtype": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D list, uint8
    input_dict = {"shape": [4, 4], "dtype": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D list with unit dimensions, int16
    input_dict = {"shape": [1, 1, 1], "dtype": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D list, complex128
    input_dict = {"shape": [3, 2], "dtype": np.dtype(np.complex128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D list, float16
    input_dict = {"shape": [2, 1, 2, 1, 2], "dtype": np.dtype(np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.ones_3"] = ones_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ones_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ones_3'.")


check_valid('jax.numpy.ones', generated_inputs['jax.numpy.ones_3'], lib="jax", suffix=3)
