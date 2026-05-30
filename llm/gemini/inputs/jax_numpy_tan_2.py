
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tan_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 2: Positive float
    list_of_inputs.append({"x": 1.0})

    # Input 3: Negative float
    list_of_inputs.append({"x": -1.0})

    # Input 4: pi / 4
    list_of_inputs.append({"x": float(np.pi / 4)})

    # Input 5: pi / 2
    list_of_inputs.append({"x": float(np.pi / 2)})

    # Input 6: Negative pi
    list_of_inputs.append({"x": float(-np.pi)})

    # Input 7: Very small positive float
    list_of_inputs.append({"x": 1e-6})

    # Input 8: Large positive float
    list_of_inputs.append({"x": 1000.0})

    # Input 9: Large negative float
    list_of_inputs.append({"x": -1000.0})

    # Input 10: pi / 3
    list_of_inputs.append({"x": float(np.pi / 3)})

    # Input 11: Negative pi / 6
    list_of_inputs.append({"x": float(-np.pi / 6)})

    return list_of_inputs

generated_inputs["jax.numpy.tan_2"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tan_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tan_2'.")


check_valid('jax.numpy.tan', generated_inputs['jax.numpy.tan_2'], lib="jax", suffix=2)
