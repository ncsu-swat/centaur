
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arange_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float range, float32
    input_dict = {
        'start': 0.0,
        'stop': 10.0,
        'step': 1.0,
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Fractional step, float64
    input_dict = {
        'start': 0.0,
        'stop': 1.0,
        'step': 0.1,
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative to positive range, int32
    input_dict = {
        'start': -10.0,
        'stop': 10.0,
        'step': 2.0,
        'dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative step (descending), float32
    input_dict = {
        'start': 10.0,
        'stop': 0.0,
        'step': -0.5,
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small range with float16
    input_dict = {
        'start': 1.0,
        'stop': 2.0,
        'step': 0.125,
        'dtype': np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values, int64
    input_dict = {
        'start': 100.0,
        'stop': 105.0,
        'step': 1.0,
        'dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative float range, float32
    input_dict = {
        'start': -5.0,
        'stop': -1.0,
        'step': 0.5,
        'dtype': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High precision step, float64
    input_dict = {
        'start': 0.0,
        'stop': 0.1,
        'step': 0.01,
        'dtype': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Descending range to negative values, int32
    input_dict = {
        'start': 20.0,
        'stop': 10.0,
        'step': -2.5,
        'dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero-centered symmetric range, float32
    input_dict = {
        'start': -0.5,
        'stop': 0.5,
        'step': 0.1,
        'dtype': np.float32
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
