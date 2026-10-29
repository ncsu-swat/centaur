
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_inputs():
    list_of_inputs = []

    # Input 1: 0-length array with bool dtype
    input_dict = {
        "shape": 0,
        "fill_value": True,
        "dtype": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: single element array with bool dtype
    input_dict = {
        "shape": 1,
        "fill_value": False,
        "dtype": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: integer dtype with True
    input_dict = {
        "shape": 5,
        "fill_value": True,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 dtype with False
    input_dict = {
        "shape": 10,
        "fill_value": False,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 dtype with True
    input_dict = {
        "shape": 3,
        "fill_value": True,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int8 dtype with True
    input_dict = {
        "shape": 7,
        "fill_value": True,
        "dtype": np.dtype(np.int8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8 dtype with False
    input_dict = {
        "shape": 2,
        "fill_value": False,
        "dtype": np.dtype(np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64 dtype with True
    input_dict = {
        "shape": 16,
        "fill_value": True,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 dtype with False
    input_dict = {
        "shape": 4,
        "fill_value": False,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint32 dtype with True
    input_dict = {
        "shape": 8,
        "fill_value": True,
        "dtype": np.dtype(np.uint32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float16 dtype with False
    input_dict = {
        "shape": 12,
        "fill_value": False,
        "dtype": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.full_3"] = full_inputs()

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
