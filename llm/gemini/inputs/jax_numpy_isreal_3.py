
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isreal_inputs():
    list_of_inputs = []

    # Input 1: Basic positive float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float
    input_dict = {"x": -2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Infinity
    input_dict = {"x": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative Infinity
    input_dict = {"x": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NaN (Not a Number)
    input_dict = {"x": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Numpy float32
    input_dict = {"x": float(np.float32(3.14))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Numpy float64
    input_dict = {"x": float(np.float64(-0.0001))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small scientific notation float
    input_dict = {"x": 1.23e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger float value
    input_dict = {"x": 987654.321}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isreal_3"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isreal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isreal_3'.")


check_valid('jax.numpy.isreal', generated_inputs['jax.numpy.isreal_3'], lib="jax", suffix=3)
