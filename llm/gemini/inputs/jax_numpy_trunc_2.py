
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def trunc_inputs():
    list_of_inputs = []

    # Input 1: Positive float with fraction
    list_of_inputs.append({"x": 3.7})

    # Input 2: Negative float with fraction
    list_of_inputs.append({"x": -3.7})

    # Input 3: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 4: Negative zero
    list_of_inputs.append({"x": -0.0})

    # Input 5: Fractional value between 0 and 1
    list_of_inputs.append({"x": 0.99999})

    # Input 6: Fractional value between -1 and 0
    list_of_inputs.append({"x": -0.99999})

    # Input 7: Exact positive integer as float
    list_of_inputs.append({"x": 42.0})

    # Input 8: Exact negative integer as float
    list_of_inputs.append({"x": -100.0})

    # Input 9: Large positive float
    list_of_inputs.append({"x": 123456.789})

    # Input 10: Large negative float
    list_of_inputs.append({"x": -987654.321})

    # Input 11: Positive infinity
    list_of_inputs.append({"x": float("inf")})

    # Input 12: Negative infinity
    list_of_inputs.append({"x": float("-inf")})

    return list_of_inputs

generated_inputs["jax.numpy.trunc_2"] = trunc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trunc_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trunc_2'.")


check_valid('jax.numpy.trunc', generated_inputs['jax.numpy.trunc_2'], lib="jax", suffix=2)
