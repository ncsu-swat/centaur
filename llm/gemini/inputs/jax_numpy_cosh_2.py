
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cosh_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 2: Small positive float
    list_of_inputs.append({"x": 0.5})

    # Input 3: Small negative float
    list_of_inputs.append({"x": -0.5})

    # Input 4: Standard positive float
    list_of_inputs.append({"x": 1.0})

    # Input 5: Standard negative float
    list_of_inputs.append({"x": -1.0})

    # Input 6: Larger positive float
    list_of_inputs.append({"x": 10.0})

    # Input 7: Larger negative float
    list_of_inputs.append({"x": -10.0})

    # Input 8: Very small float
    list_of_inputs.append({"x": 1e-6})

    # Input 9: Pi float
    list_of_inputs.append({"x": 3.141592653589793})

    # Input 10: Negative Pi float
    list_of_inputs.append({"x": -3.141592653589793})

    # Input 11: Large magnitude float
    list_of_inputs.append({"x": 50.0})

    # Input 12: Negative large magnitude float
    list_of_inputs.append({"x": -50.0})

    return list_of_inputs

generated_inputs["jax.numpy.cosh_2"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cosh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cosh_2'.")


check_valid('jax.numpy.cosh', generated_inputs['jax.numpy.cosh_2'], lib="jax", suffix=2)
