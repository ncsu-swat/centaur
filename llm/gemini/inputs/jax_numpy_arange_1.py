
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arange_inputs():
    list_of_inputs = []

    # Input 1: Basic positive range with int32 dtype
    input_dict = {
        "start": 0,
        "stop": 10,
        "step": 1,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-zero start and step > 1 with int64 dtype
    input_dict = {
        "start": 5,
        "stop": 25,
        "step": 5,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative start to positive stop with float32 dtype
    input_dict = {
        "start": -10,
        "stop": 10,
        "step": 2,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Decreasing range with negative step
    input_dict = {
        "start": 10,
        "stop": 0,
        "step": -1,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Range with both negative start and stop
    input_dict = {
        "start": -20,
        "stop": -5,
        "step": 3,
        "dtype": np.dtype(np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large step with float64 dtype
    input_dict = {
        "start": 0,
        "stop": 100,
        "step": 10,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty range (start == stop)
    input_dict = {
        "start": 0,
        "stop": 0,
        "step": 1,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative step across zero
    input_dict = {
        "start": 100,
        "stop": -100,
        "step": -20,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Unsigned integer dtype
    input_dict = {
        "start": 1,
        "stop": 8,
        "step": 2,
        "dtype": np.dtype(np.uint32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 8-bit integer dtype
    input_dict = {
        "start": -50,
        "stop": 50,
        "step": 25,
        "dtype": np.dtype(np.int8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Unsigned 8-bit integer dtype
    input_dict = {
        "start": 0,
        "stop": 5,
        "step": 1,
        "dtype": np.dtype(np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Complex dtype
    input_dict = {
        "start": -10,
        "stop": -2,
        "step": 1,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arange_1"] = arange_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arange_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arange_1'.")


check_valid('jax.numpy.arange', generated_inputs['jax.numpy.arange_1'], lib="jax", suffix=1)
