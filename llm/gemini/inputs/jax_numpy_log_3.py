
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_log_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    list_of_inputs.append({"x": 1.0})

    # Input 2: Small positive float
    list_of_inputs.append({"x": 0.5})

    # Input 3: Approximation of e
    list_of_inputs.append({"x": 2.718281828459045})

    # Input 4: Integer-valued float
    list_of_inputs.append({"x": 10.0})

    # Input 5: Large positive float
    list_of_inputs.append({"x": 10000.0})

    # Input 6: Extremely small positive float
    list_of_inputs.append({"x": 1e-8})

    # Input 7: Double precision float representation of pi
    list_of_inputs.append({"x": 3.141592653589793})

    # Input 8: Floating point value greater than 1
    list_of_inputs.append({"x": 50.5})

    # Input 9: Zero (results in -inf, but valid float input)
    list_of_inputs.append({"x": 0.0})

    # Input 10: Negative float (results in nan, but valid float input)
    list_of_inputs.append({"x": -2.5})

    return list_of_inputs

generated_inputs["jax.numpy.log_3"] = jax_numpy_log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log_3'.")


check_valid('jax.numpy.log', generated_inputs['jax.numpy.log_3'], lib="jax", suffix=3)
