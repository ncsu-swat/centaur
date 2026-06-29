
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_le_inputs():
    list_of_inputs = []

    # Input 1: positive floats, x < y
    input_dict = {"x": 1.5, "y": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative floats, x > y
    input_dict = {"x": -1.0, "y": -2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: equal floats
    input_dict = {"x": 3.14, "y": 3.14}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: zeros with different signs
    input_dict = {"x": 0.0, "y": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: large floats
    input_dict = {"x": 1e10, "y": 1e11}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: small floats
    input_dict = {"x": 1e-10, "y": 1e-9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: infinity comparison
    input_dict = {"x": float('-inf'), "y": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: positive infinity equal
    input_dict = {"x": float('inf'), "y": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: mixed sign
    input_dict = {"x": -5.7, "y": 5.7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NaN comparison
    input_dict = {"x": float('nan'), "y": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.le_3"] = jax_lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_3'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_3'], lib="jax", suffix=3)
