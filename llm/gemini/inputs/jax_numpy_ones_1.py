
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ones_inputs():
    list_of_inputs = []

    # Input 1: shape=0, float32
    input_dict = {"shape": 0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: shape=1, int32
    input_dict = {"shape": 1, "dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: shape=2, bool
    input_dict = {"shape": 2, "dtype": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: shape=5, float64
    input_dict = {"shape": 5, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: shape=7, int16
    input_dict = {"shape": 7, "dtype": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: shape=10, int64
    input_dict = {"shape": 10, "dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: shape=16, complex64
    input_dict = {"shape": 16, "dtype": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: shape=32, uint8
    input_dict = {"shape": 32, "dtype": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: shape=64, int8
    input_dict = {"shape": 64, "dtype": np.dtype(np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: shape=100, complex128
    input_dict = {"shape": 100, "dtype": np.dtype(np.complex128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: shape=128, uint32
    input_dict = {"shape": 128, "dtype": np.dtype(np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.ones_1"] = ones_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ones_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ones_1'.")


check_valid('jax.numpy.ones', generated_inputs['jax.numpy.ones_1'], lib="jax", suffix=1)
