
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arccosh_inputs():
    list_of_inputs = []

    # Input 1: Boundary value 1.0
    list_of_inputs.append({"x": 1.0})

    # Input 2: Small value > 1.0
    list_of_inputs.append({"x": 1.1})

    # Input 3: Standard float value
    list_of_inputs.append({"x": 1.5})

    # Input 4: Standard float value
    list_of_inputs.append({"x": 2.0})

    # Input 5: Intermediate float value
    list_of_inputs.append({"x": 5.0})

    # Input 6: Larger float value
    list_of_inputs.append({"x": 10.0})

    # Input 7: Very large float value
    list_of_inputs.append({"x": 100.0})

    # Input 8: Float with decimals
    list_of_inputs.append({"x": 12.34})

    # Input 9: Value close to 1
    list_of_inputs.append({"x": 1.0001})

    # Input 10: Another float value
    list_of_inputs.append({"x": 50.5})

    return list_of_inputs

generated_inputs["jax.numpy.arccosh_2"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_2'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_2'], lib="jax", suffix=2)
