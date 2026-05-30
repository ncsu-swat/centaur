
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: negative x1, positive x2
    list_of_inputs.append({"x1": -2.5, "x2": 0.5})

    # Input 2: zero x1, positive x2
    list_of_inputs.append({"x1": 0.0, "x2": 0.5})

    # Input 3: positive x1, positive x2
    list_of_inputs.append({"x1": 3.5, "x2": 0.5})

    # Input 4: negative x1, negative x2
    list_of_inputs.append({"x1": -1.2, "x2": -0.8})

    # Input 5: zero x1, negative x2
    list_of_inputs.append({"x1": 0.0, "x2": -1.5})

    # Input 6: positive x1, negative x2
    list_of_inputs.append({"x1": 5.0, "x2": -2.0})

    # Input 7: zero x1, zero x2
    list_of_inputs.append({"x1": 0.0, "x2": 0.0})

    # Input 8: large positive x1, positive x2
    list_of_inputs.append({"x1": 1e6, "x2": 1.0})

    # Input 9: large negative x1, positive x2
    list_of_inputs.append({"x1": -1e6, "x2": 1.0})

    # Input 10: small positive x1, positive x2
    list_of_inputs.append({"x1": 1e-6, "x2": 0.5})

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_6"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_6'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_6'], lib="jax", suffix=6)
