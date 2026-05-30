
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def modf_inputs():
    list_of_inputs = []

    floats = [
        4.8, 
        -3.4, 
        0.0, 
        12345.6789, 
        -0.0001, 
        1e10 + 0.5, 
        float('inf'), 
        float('-inf'), 
        float('nan'), 
        -123.456
    ]
    
    for f in floats:
        input_dict = {
            "x": f,
            "out": None
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.modf_2"] = modf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.modf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.modf_2'.")


check_valid('jax.numpy.modf', generated_inputs['jax.numpy.modf_2'], lib="jax", suffix=2)
