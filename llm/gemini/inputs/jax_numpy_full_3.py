
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_full_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean fill, boolean output
    input_dict = {
        "shape": 5,
        "fill_value": True,
        "dtype": np.dtype('bool')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: False fill, integer output
    input_dict = {
        "shape": 3,
        "fill_value": False,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: True fill, float32 output
    input_dict = {
        "shape": 10,
        "fill_value": True,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small shape, False fill, int8 output
    input_dict = {
        "shape": 1,
        "fill_value": False,
        "dtype": np.dtype('int8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: True fill, uint8 output
    input_dict = {
        "shape": 8,
        "fill_value": True,
        "dtype": np.dtype('uint8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: True fill, int16 output
    input_dict = {
        "shape": 12,
        "fill_value": True,
        "dtype": np.dtype('int16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger shape, False fill, float64 output
    input_dict = {
        "shape": 20,
        "fill_value": False,
        "dtype": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: True fill, int64 output
    input_dict = {
        "shape": 15,
        "fill_value": True,
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: False fill, complex64 output
    input_dict = {
        "shape": 2,
        "fill_value": False,
        "dtype": np.dtype('complex64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: True fill, complex128 output
    input_dict = {
        "shape": 7,
        "fill_value": True,
        "dtype": np.dtype('complex128')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.full_3"] = jax_numpy_full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_3'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_3'], lib="jax", suffix=3)
