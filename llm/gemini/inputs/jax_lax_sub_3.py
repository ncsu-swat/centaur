
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_sub_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    list_of_inputs.append({"x": 5.0, "y": 3.0})

    # Input 2: Negative and positive float
    list_of_inputs.append({"x": -1.5, "y": 2.5})

    # Input 3: Zeroes
    list_of_inputs.append({"x": 0.0, "y": 0.0})

    # Input 4: Large floats
    list_of_inputs.append({"x": 1e6, "y": 5e5})

    # Input 5: Small fractional floats
    list_of_inputs.append({"x": 1e-5, "y": 2e-5})

    # Input 6: Pi and E approximations
    list_of_inputs.append({"x": 3.14159265, "y": 2.71828182})

    # Input 7: Both negative floats
    list_of_inputs.append({"x": -100.75, "y": -50.25})

    # Input 8: Floats with many decimal places
    list_of_inputs.append({"x": 0.123456789, "y": 0.987654321})

    # Input 9: Large difference in scale
    list_of_inputs.append({"x": 10000.0, "y": 0.0001})

    # Input 10: Equal floats
    list_of_inputs.append({"x": 42.42, "y": 42.42})

    return list_of_inputs

generated_inputs["jax.lax.sub_3"] = jax_lax_sub_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sub_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sub_3'.")


check_valid('jax.lax.sub', generated_inputs['jax.lax.sub_3'], lib="jax", suffix=3)
