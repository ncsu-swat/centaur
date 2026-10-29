
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arange_inputs():
    list_of_inputs = []

    # Input 1: Basic positive range with float32
    input_dict = {
        "start": 0.0,
        "stop": 10.0,
        "step": 1.0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Fractional step size with float64
    input_dict = {
        "start": 1.5,
        "stop": 5.5,
        "step": 0.5,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative step size (decreasing range)
    input_dict = {
        "start": 10.0,
        "stop": 0.0,
        "step": -1.0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative range to positive range
    input_dict = {
        "start": -5.0,
        "stop": 5.0,
        "step": 2.0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All negative values with negative step
    input_dict = {
        "start": -2.0,
        "stop": -10.0,
        "step": -1.5,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small step size with float16
    input_dict = {
        "start": 0.0,
        "stop": 1.0,
        "step": 0.1,
        "dtype": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer dtype output from float inputs
    input_dict = {
        "start": 0.0,
        "stop": 20.0,
        "step": 2.0,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 dtype with negative values
    input_dict = {
        "start": -10.0,
        "stop": 0.0,
        "step": 1.0,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large step size resulting in a short array
    input_dict = {
        "start": 0.0,
        "stop": 100.0,
        "step": 30.0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty range (start >= stop with positive step)
    input_dict = {
        "start": 5.0,
        "stop": 2.0,
        "step": 1.0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: uint32 dtype
    input_dict = {
        "start": 1.0,
        "stop": 8.0,
        "step": 2.0,
        "dtype": np.dtype(np.uint32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arange_2"] = arange_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arange_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arange_2'.")


check_valid('jax.numpy.arange', generated_inputs['jax.numpy.arange_2'], lib="jax", suffix=2)
