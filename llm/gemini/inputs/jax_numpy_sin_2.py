
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sin_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive angle (pi / 2)
    input_dict = {"x": float(np.pi / 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive angle (pi)
    input_dict = {"x": float(np.pi)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative angle (-pi / 2)
    input_dict = {"x": float(-np.pi / 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Full circle (2 * pi)
    input_dict = {"x": float(2 * np.pi)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Arbitrary positive float
    input_dict = {"x": 1.2345}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Arbitrary negative float
    input_dict = {"x": -45.67}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small positive float
    input_dict = {"x": 1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large float
    input_dict = {"x": 1e6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Special float (inf)
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Special float (nan)
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative zero
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.sin_2"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sin_2'.")


check_valid('jax.numpy.sin', generated_inputs['jax.numpy.sin_2'], lib="jax", suffix=2)
