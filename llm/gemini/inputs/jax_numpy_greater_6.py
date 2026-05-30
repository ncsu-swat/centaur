
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_greater_inputs():
    list_of_inputs = []

    # Input 1: positive floats
    list_of_inputs.append({"x": 5.0, "y": 2.0})

    # Input 2: negative and positive floats
    list_of_inputs.append({"x": -1.5, "y": 3.2})

    # Input 3: equal floats
    list_of_inputs.append({"x": 0.0, "y": 0.0})

    # Input 4: large floats
    list_of_inputs.append({"x": 1e10, "y": 1e9})

    # Input 5: negative floats
    list_of_inputs.append({"x": -10.5, "y": -20.5})

    # Input 6: mathematical approximations
    list_of_inputs.append({"x": 3.14159, "y": 2.71828})

    # Input 7: small floats
    list_of_inputs.append({"x": 1e-6, "y": 2e-6})

    # Input 8: infinity and large float
    list_of_inputs.append({"x": float('inf'), "y": 1e20})

    # Input 9: negative float and negative infinity
    list_of_inputs.append({"x": -1e20, "y": float('-inf')})

    # Input 10: very close floats
    list_of_inputs.append({"x": 1.23456, "y": 1.23455})

    return list_of_inputs

generated_inputs["jax.numpy.greater_6"] = jax_numpy_greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_6'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_6'], lib="jax", suffix=6)
