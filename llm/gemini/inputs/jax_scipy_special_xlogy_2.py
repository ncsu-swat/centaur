
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def xlogy_inputs():
    list_of_inputs = []

    # Input 1: x=0.0, y=0.0 (special case, should return 0)
    list_of_inputs.append({"x": 0.0, "y": 0.0})

    # Input 2: x=0.0, y=5.5
    list_of_inputs.append({"x": 0.0, "y": 5.5})

    # Input 3: x=1.0, y=1.0
    list_of_inputs.append({"x": 1.0, "y": 1.0})

    # Input 4: x=2.5, y=10.0
    list_of_inputs.append({"x": 2.5, "y": 10.0})

    # Input 5: x=-3.0, y=2.0 (negative x value)
    list_of_inputs.append({"x": -3.0, "y": 2.0})

    # Input 6: x=0.5, y=0.5
    list_of_inputs.append({"x": 0.5, "y": 0.5})

    # Input 7: x=1e-5, y=1e5
    list_of_inputs.append({"x": 1e-5, "y": 1e5})

    # Input 8: x=100.0, y=0.1
    list_of_inputs.append({"x": 100.0, "y": 0.1})

    # Input 9: x=-1.5, y=0.01
    list_of_inputs.append({"x": -1.5, "y": 0.01})

    # Input 10: x=0.0, y=-2.0
    list_of_inputs.append({"x": 0.0, "y": -2.0})

    # Input 11: x=4.2, y=4.2
    list_of_inputs.append({"x": 4.2, "y": 4.2})

    return list_of_inputs

generated_inputs["jax.scipy.special.xlogy_2"] = xlogy_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.xlogy_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.xlogy_2'.")


check_valid('jax.scipy.special.xlogy', generated_inputs['jax.scipy.special.xlogy_2'], lib="jax", suffix=2)
