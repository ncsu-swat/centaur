
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({"x": 0})

    # Input 2
    list_of_inputs.append({"x": 1})

    # Input 3
    list_of_inputs.append({"x": -1})

    # Input 4
    list_of_inputs.append({"x": 100})

    # Input 5
    list_of_inputs.append({"x": -100})

    # Input 6
    list_of_inputs.append({"x": 5000})

    # Input 7
    list_of_inputs.append({"x": -5000})

    # Input 8
    list_of_inputs.append({"x": 1234567})

    # Input 9
    list_of_inputs.append({"x": -1234567})

    # Input 10
    list_of_inputs.append({"x": 2147483647})

    # Input 11
    list_of_inputs.append({"x": -2147483648})

    return list_of_inputs

generated_inputs["jax.numpy.iscomplexobj_3"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_3'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_3'], lib="jax", suffix=3)
