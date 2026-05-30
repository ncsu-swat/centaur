
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def float_power_inputs():
    list_of_inputs = []

    # Input 1: Simple positive floats
    list_of_inputs.append({"x": 2.0, "y": 3.0})

    # Input 2: Negative exponent
    list_of_inputs.append({"x": 5.0, "y": -1.0})

    # Input 3: Fractional exponent (square root)
    list_of_inputs.append({"x": 9.0, "y": 0.5})

    # Input 4: Negative base, odd integer exponent
    list_of_inputs.append({"x": -2.0, "y": 3.0})

    # Input 5: Zero base with positive exponent
    list_of_inputs.append({"x": 0.0, "y": 5.0})

    # Input 6: Fractional base and fractional exponent
    list_of_inputs.append({"x": 1.5, "y": 2.5})

    # Input 7: Base > 1, negative exponent
    list_of_inputs.append({"x": 10.0, "y": -2.0})

    # Input 8: Negative base, even integer exponent
    list_of_inputs.append({"x": -3.0, "y": 2.0})

    # Input 9: Non-zero base, zero exponent
    list_of_inputs.append({"x": 4.0, "y": 0.0})

    # Input 10: Decimal base and negative fractional exponent
    list_of_inputs.append({"x": 0.25, "y": -0.5})

    # Input 11: Very small base
    list_of_inputs.append({"x": 1e-5, "y": 2.0})

    return list_of_inputs

generated_inputs["jax.numpy.float_power_2"] = float_power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.float_power_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.float_power_2'.")


check_valid('jax.numpy.float_power', generated_inputs['jax.numpy.float_power_2'], lib="jax", suffix=2)
