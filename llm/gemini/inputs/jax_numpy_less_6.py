
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_less_inputs():
    list_of_inputs = []

    # Input 1: positive floats
    list_of_inputs.append({"x": 1.0, "y": 2.0})

    # Input 2: positive and negative floats
    list_of_inputs.append({"x": 5.5, "y": -1.2})

    # Input 3: equal negative floats
    list_of_inputs.append({"x": -3.14, "y": -3.14})

    # Input 4: zeroes
    list_of_inputs.append({"x": 0.0, "y": 0.0})

    # Input 5: signed zeroes
    list_of_inputs.append({"x": -0.0, "y": 0.0})

    # Input 6: positive and negative infinities
    list_of_inputs.append({"x": float('inf'), "y": float('-inf')})

    # Input 7: small scale floats
    list_of_inputs.append({"x": 1e-5, "y": 1e-4})

    # Input 8: large scale floats
    list_of_inputs.append({"x": 1.23e10, "y": 1.24e10})

    # Input 9: NaN float comparison
    list_of_inputs.append({"x": float('nan'), "y": 1.0})

    # Input 10: close negative floats
    list_of_inputs.append({"x": -9999.99, "y": -9999.98})

    return list_of_inputs

generated_inputs["jax.numpy.less_6"] = jax_numpy_less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_6'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_6'], lib="jax", suffix=6)
