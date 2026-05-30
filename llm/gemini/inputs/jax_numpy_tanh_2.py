
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tanh_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    list_of_inputs.append({"x": 1.0})

    # Input 2: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 3: Standard negative float
    list_of_inputs.append({"x": -1.0})

    # Input 4: Small positive float
    list_of_inputs.append({"x": 0.25})

    # Input 5: Small negative float
    list_of_inputs.append({"x": -0.25})

    # Input 6: Large positive float
    list_of_inputs.append({"x": 15.0})

    # Input 7: Large negative float
    list_of_inputs.append({"x": -15.0})

    # Input 8: Float representing pi
    list_of_inputs.append({"x": 3.141592653589793})

    # Input 9: Very small float close to zero
    list_of_inputs.append({"x": 1e-6})

    # Input 10: Negative float representing -pi
    list_of_inputs.append({"x": -3.141592653589793})

    return list_of_inputs

generated_inputs["jax.numpy.tanh_2"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tanh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tanh_2'.")


check_valid('jax.numpy.tanh', generated_inputs['jax.numpy.tanh_2'], lib="jax", suffix=2)
