
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def radians_inputs():
    list_of_inputs = []

    # Input 1: 1D array of standard float32 angles
    input_dict = {
        "x": np.array([0.0, 30.0, 45.0, 60.0, 90.0, 180.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with negative float64 angles
    input_dict = {
        "x": np.array([-360.0, -180.0, -90.0, -45.0, 0.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single element 1D array
    input_dict = {"x": np.array([180.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of int32 values
    input_dict = {"x": np.array([0, 90, 180, 270, 360], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of negative int64 values
    input_dict = {"x": np.array([-360, -270, -180, -90, 0], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with special float values (nan, inf)
    input_dict = {
        "x": np.array([0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with large angle values
    input_dict = {
        "x": np.array([720.0, 1080.0, 1440.0, -720.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with small angle values
    input_dict = {
        "x": np.array([1e-4, -1e-4, 1e-6, -1e-6], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array of booleans
    input_dict = {"x": np.array([True, False, True, True], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Finely-spaced 1D array of angles
    input_dict = {"x": np.linspace(0.0, 360.0, 37, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array of fractional angles
    input_dict = {
        "x": np.array([12.5, 37.3, 89.9, 179.9, 359.1], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.radians_3"] = radians_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.radians_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.radians_3'.")


check_valid('jax.numpy.radians', generated_inputs['jax.numpy.radians_3'], lib="jax", suffix=3)
