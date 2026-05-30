
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def real_inputs():
    list_of_inputs = []

    # Input 1: Simple positive float
    input_dict = {"val": 3.14}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float
    input_dict = {"val": -10.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero float
    input_dict = {"val": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative zero float
    input_dict = {"val": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive float
    input_dict = {"val": 1e15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Very small float
    input_dict = {"val": 1e-15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Positive infinity as float
    input_dict = {"val": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative infinity as float
    input_dict = {"val": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NaN as float
    input_dict = {"val": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float represented by np.float32
    input_dict = {"val": float(np.float32(2.71828))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Float represented by np.float64
    input_dict = {"val": float(np.float64(-0.57721))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.real_3"] = real_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.real_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.real_3'.")


check_valid('jax.numpy.real', generated_inputs['jax.numpy.real_3'], lib="jax", suffix=3)
