
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reciprocal_inputs():
    list_of_inputs = []

    # Input 1: positive float
    list_of_inputs.append({"x": 2.0})

    # Input 2: negative float
    list_of_inputs.append({"x": -5.0})

    # Input 3: decimal float less than 1
    list_of_inputs.append({"x": 0.25})

    # Input 4: negative decimal float
    list_of_inputs.append({"x": -0.125})

    # Input 5: large float
    list_of_inputs.append({"x": 10000.0})

    # Input 6: small float
    list_of_inputs.append({"x": 1e-6})

    # Input 7: Euler's number as float
    list_of_inputs.append({"x": float(np.e)})

    # Input 8: Pi as float
    list_of_inputs.append({"x": float(np.pi)})

    # Input 9: Float infinity
    list_of_inputs.append({"x": float('inf')})

    # Input 10: Float negative infinity
    list_of_inputs.append({"x": float('-inf')})

    # Input 11: Zero float
    list_of_inputs.append({"x": 0.0})

    return list_of_inputs

generated_inputs["jax.numpy.reciprocal_2"] = reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reciprocal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reciprocal_2'.")


check_valid('jax.numpy.reciprocal', generated_inputs['jax.numpy.reciprocal_2'], lib="jax", suffix=2)
