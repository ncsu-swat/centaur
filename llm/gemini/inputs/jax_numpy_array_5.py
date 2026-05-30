
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def array_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "object": 1.5,
        "dtype": np.float32,
        "copy": True,
        "order": "K",
        "ndmin": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "object": -3.14,
        "dtype": np.float64,
        "copy": False,
        "order": "K",
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "object": 0.0,
        "dtype": np.int32,
        "copy": True,
        "order": "K",
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "object": float('inf'),
        "dtype": np.float32,
        "copy": True,
        "order": "K",
        "ndmin": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "object": -0.0001,
        "dtype": np.float16,
        "copy": False,
        "order": "K",
        "ndmin": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "object": 42.0,
        "dtype": np.complex64,
        "copy": True,
        "order": "K",
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "object": float('nan'),
        "dtype": np.float32,
        "copy": False,
        "order": "K",
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "object": 1234.5678,
        "dtype": np.float64,
        "copy": True,
        "order": "K",
        "ndmin": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "object": -1.0,
        "dtype": np.int64,
        "copy": False,
        "order": "K",
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "object": 2.718,
        "dtype": np.float32,
        "copy": True,
        "order": "K",
        "ndmin": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.array_5"] = array_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_5'.")


check_valid('jax.numpy.array', generated_inputs['jax.numpy.array_5'], lib="jax", suffix=5)
