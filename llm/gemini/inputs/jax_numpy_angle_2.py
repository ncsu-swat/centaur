
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def angle_inputs():
    list_of_inputs = []

    # Input 1: positive float, deg=False
    input_dict = {"z": 1.0, "deg": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float, deg=True
    input_dict = {"z": -1.0, "deg": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: positive zero, deg=False
    input_dict = {"z": 0.0, "deg": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: negative zero, deg=True
    input_dict = {"z": -0.0, "deg": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: positive arbitrary float, deg=False
    input_dict = {"z": 3.141592653589793, "deg": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: negative arbitrary float, deg=True
    input_dict = {"z": -2.718281828459045, "deg": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: very small positive float, deg=True
    input_dict = {"z": 1e-12, "deg": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: very large negative float, deg=False
    input_dict = {"z": -1e12, "deg": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: positive infinity, deg=False
    input_dict = {"z": float("inf"), "deg": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative infinity, deg=True
    input_dict = {"z": float("-inf"), "deg": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: fractional positive float, deg=True
    input_dict = {"z": 42.5, "deg": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: fractional negative float, deg=False
    input_dict = {"z": -100.25, "deg": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.angle_2"] = angle_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.angle_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.angle_2'.")


check_valid('jax.numpy.angle', generated_inputs['jax.numpy.angle_2'], lib="jax", suffix=2)
