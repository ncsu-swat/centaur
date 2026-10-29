
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_inputs():
    list_of_inputs = []

    # Input 1: Standard float array
    input_dict = {
        "shape": 5,
        "fill_value": 3.14,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero shape (empty array)
    input_dict = {
        "shape": 0,
        "fill_value": 0.0,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float converted to int dtype
    input_dict = {
        "shape": 1,
        "fill_value": -2.5,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float converted to boolean dtype
    input_dict = {
        "shape": 10,
        "fill_value": 1.0,
        "dtype": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Infinity fill value
    input_dict = {
        "shape": 20,
        "fill_value": float("inf"),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative infinity fill value
    input_dict = {
        "shape": 3,
        "fill_value": float("-inf"),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NaN fill value
    input_dict = {
        "shape": 7,
        "fill_value": float("nan"),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float converted to complex64
    input_dict = {
        "shape": 2,
        "fill_value": 100.5,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative zero with float16 dtype
    input_dict = {
        "shape": 16,
        "fill_value": -0.0,
        "dtype": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float converted to unsigned int dtype
    input_dict = {
        "shape": 4,
        "fill_value": 42.0,
        "dtype": np.dtype(np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Small float to complex128
    input_dict = {
        "shape": 8,
        "fill_value": 1e-4,
        "dtype": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Float to int64
    input_dict = {
        "shape": 12,
        "fill_value": -999.0,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.full_2"] = full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_2'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_2'], lib="jax", suffix=2)
