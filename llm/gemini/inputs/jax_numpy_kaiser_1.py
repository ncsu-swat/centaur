
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def kaiser_inputs():
    list_of_inputs = []

    # Input 1: Zero window size, zero beta
    input_dict = {
        "M": int(0),
        "beta": float(0.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Size 1, positive beta
    input_dict = {
        "M": int(1),
        "beta": float(5.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Size 10, zero beta (equivalent to rectangular window)
    input_dict = {
        "M": int(10),
        "beta": float(0.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Size 5, beta = 14.0 (high sidelobe attenuation)
    input_dict = {
        "M": int(5),
        "beta": float(14.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Size 12, beta = 6.0
    input_dict = {
        "M": int(12),
        "beta": float(6.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Size 50, beta = 8.6
    input_dict = {
        "M": int(50),
        "beta": float(8.6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Size 100, beta = 10.0
    input_dict = {
        "M": int(100),
        "beta": float(10.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Size 256, beta = 5.0
    input_dict = {
        "M": int(256),
        "beta": float(5.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Size 3, negative beta
    input_dict = {
        "M": int(3),
        "beta": float(-1.5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Size 1000, large beta
    input_dict = {
        "M": int(1000),
        "beta": float(20.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.kaiser_1"] = kaiser_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.kaiser_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.kaiser_1'.")


check_valid('jax.numpy.kaiser', generated_inputs['jax.numpy.kaiser_1'], lib="jax", suffix=1)
