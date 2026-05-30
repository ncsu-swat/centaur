
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arctan_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0})

    # Input 2: One
    list_of_inputs.append({"x": 1})

    # Input 3: Negative one
    list_of_inputs.append({"x": -1})

    # Input 4: Small positive integer
    list_of_inputs.append({"x": 5})

    # Input 5: Small negative integer
    list_of_inputs.append({"x": -5})

    # Input 6: Medium positive integer
    list_of_inputs.append({"x": 42})

    # Input 7: Medium negative integer
    list_of_inputs.append({"x": -42})

    # Input 8: Larger positive integer
    list_of_inputs.append({"x": 100})

    # Input 9: Larger negative integer
    list_of_inputs.append({"x": -100})

    # Input 10: Very large integer
    list_of_inputs.append({"x": 10000})

    return list_of_inputs

generated_inputs["jax.numpy.arctan_3"] = arctan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctan_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctan_3'.")


check_valid('jax.numpy.arctan', generated_inputs['jax.numpy.arctan_3'], lib="jax", suffix=3)
