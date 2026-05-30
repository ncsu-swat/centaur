
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arctan_inputs():
    list_of_inputs = []
    
    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Positive float (1.0)
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Negative float (-1.0)
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Large positive float
    input_dict = {"x": 1e5}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Large negative float
    input_dict = {"x": -1e5}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Very small positive float
    input_dict = {"x": 1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Very small negative float
    input_dict = {"x": -1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Positive infinity
    input_dict = {"x": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Negative infinity
    input_dict = {"x": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: NaN (Not a Number)
    input_dict = {"x": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Common value approximation (sqrt(3))
    input_dict = {"x": 1.7320508}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arctan_2"] = arctan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctan_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctan_2'.")


check_valid('jax.numpy.arctan', generated_inputs['jax.numpy.arctan_2'], lib="jax", suffix=2)
