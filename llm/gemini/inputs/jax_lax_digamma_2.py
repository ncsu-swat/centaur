
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.lax

def digamma_inputs():
    list_of_inputs = []

    # Input 1: Standard positive integer value as float
    list_of_inputs.append({"x": 1.0})

    # Input 2: Fractional positive value
    list_of_inputs.append({"x": 0.5})

    # Input 3: Larger positive value
    list_of_inputs.append({"x": 2.5})

    # Input 4: Negative fractional value (avoiding non-positive integers)
    list_of_inputs.append({"x": -0.5})

    # Input 5: Small positive value
    list_of_inputs.append({"x": 0.001})

    # Input 6: Large positive float
    list_of_inputs.append({"x": 100.0})

    # Input 7: Negative non-integer float
    list_of_inputs.append({"x": -1.25})

    # Input 8: Another negative non-integer float
    list_of_inputs.append({"x": -5.7})

    # Input 9: Transcendental float
    list_of_inputs.append({"x": 3.141592653589793})

    # Input 10: Very large float
    list_of_inputs.append({"x": 1e4})

    # Input 11: Small negative float
    list_of_inputs.append({"x": -0.01})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.lax.digamma_2"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.digamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.digamma_2'.")


check_valid('jax.lax.digamma', generated_inputs['jax.lax.digamma_2'], lib="jax", suffix=2)
