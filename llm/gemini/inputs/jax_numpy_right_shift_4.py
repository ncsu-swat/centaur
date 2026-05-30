
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_right_shift_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({"x1": 16, "x2": 2})

    # Input 2
    list_of_inputs.append({"x1": 0, "x2": 1})

    # Input 3
    list_of_inputs.append({"x1": 1024, "x2": 10})

    # Input 4
    list_of_inputs.append({"x1": 1, "x2": 0})

    # Input 5
    list_of_inputs.append({"x1": -8, "x2": 1})

    # Input 6
    list_of_inputs.append({"x1": 1000000, "x2": 5})

    # Input 7
    list_of_inputs.append({"x1": -1024, "x2": 3})

    # Input 8
    list_of_inputs.append({"x1": 7, "x2": 2})

    # Input 9
    list_of_inputs.append({"x1": 123456, "x2": 8})

    # Input 10
    list_of_inputs.append({"x1": -1, "x2": 4})

    return list_of_inputs

generated_inputs["jax.numpy.right_shift_4"] = jax_numpy_right_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.right_shift_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.right_shift_4'.")


check_valid('jax.numpy.right_shift', generated_inputs['jax.numpy.right_shift_4'], lib="jax", suffix=4)
