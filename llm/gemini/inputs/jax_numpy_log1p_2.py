
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log1p_inputs():
    list_of_inputs = []

    # Input 1: zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive unit
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: small positive value
    input_dict = {"x": 1e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: very small positive value
    input_dict = {"x": 1e-15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative value in valid range (-1, 0)
    input_dict = {"x": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: negative value close to -1
    input_dict = {"x": -0.9999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: small negative value
    input_dict = {"x": -1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: moderate positive value
    input_dict = {"x": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large positive value
    input_dict = {"x": 1e8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: value yielding 1 (e - 1)
    input_dict = {"x": float(np.e - 1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: fractional float
    input_dict = {"x": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: another positive float
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.log1p_2"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log1p_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log1p_2'.")


check_valid('jax.numpy.log1p', generated_inputs['jax.numpy.log1p_2'], lib="jax", suffix=2)
