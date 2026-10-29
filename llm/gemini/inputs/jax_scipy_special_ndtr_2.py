
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def ndtr_inputs():
    list_of_inputs = []

    # Input 1: zero
    input_dict = {"x": float(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive standard deviation
    input_dict = {"x": float(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative standard deviation
    input_dict = {"x": float(-1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: moderate positive value
    input_dict = {"x": float(2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: moderate negative value
    input_dict = {"x": float(-2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: small positive fraction
    input_dict = {"x": float(0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: small negative fraction
    input_dict = {"x": float(-0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large positive value (near 1.0)
    input_dict = {"x": float(6.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large negative value (near 0.0)
    input_dict = {"x": float(-6.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: very small positive value
    input_dict = {"x": float(1e-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: arbitrary positive floating value
    input_dict = {"x": float(3.141592653589793)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: arbitrary negative floating value
    input_dict = {"x": float(-3.141592653589793)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.ndtr_2"] = ndtr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.ndtr_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.ndtr_2'.")


check_valid('jax.scipy.special.ndtr', generated_inputs['jax.scipy.special.ndtr_2'], lib="jax", suffix=2)
