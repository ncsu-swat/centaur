
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ones_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    input_dict = {"shape": [5], "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32
    input_dict = {"shape": [2, 3], "dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D bool
    input_dict = {"shape": [2, 2, 2], "dtype": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64
    input_dict = {"shape": [10, 10], "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D int16
    input_dict = {"shape": [1, 2, 1, 3], "dtype": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D uint8
    input_dict = {"shape": [100], "dtype": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D complex64
    input_dict = {"shape": [3, 4, 5], "dtype": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int64
    input_dict = {"shape": [8, 8], "dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float16
    input_dict = {"shape": [2, 1, 4], "dtype": np.dtype(np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D with zero dimension (empty array)
    input_dict = {"shape": [0, 5], "dtype": np.dtype(np.float32)}
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
