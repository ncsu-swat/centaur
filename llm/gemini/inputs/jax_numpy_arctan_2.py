
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def arctan_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive 1.0 (pi/4)
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative 1.0 (-pi/4)
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive value
    input_dict = {"x": 1e5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative value
    input_dict = {"x": -1e5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small positive value near zero
    input_dict = {"x": 1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small negative value near zero
    input_dict = {"x": -1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive square root of 3 (pi/3)
    input_dict = {"x": float(np.sqrt(3.0))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Arbitrary negative float
    input_dict = {"x": -2.71828}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Arbitrary positive float
    input_dict = {"x": 3.1415926535}
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
