
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def radians_inputs():
    list_of_inputs = []

    # Input 1: 1D array of common angles (float32)
    x = np.array([0.0, 90.0, 180.0, 270.0, 360.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with negative and positive values (float64)
    x = np.array([-180.0, -90.0, 0.0, 45.5, 360.0], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of integers
    x = np.array([-180, -90, 0, 90, 180], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty 1D array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element 1D array
    x = np.array([180.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of angles (float32)
    x = np.array([[0.0, 30.0, 45.0], [60.0, 90.0, 180.0]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of integers (int64)
    x = np.array([[-360, -180], [180, 360]], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array of angles
    x = np.array([
        [[0.0, 90.0], [180.0, 270.0]],
        [[360.0, 450.0], [-90.0, -180.0]]
    ], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array containing special float values (nan, inf)
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with large angle values
    x = np.array([1e5, -1e5, 720.0, -720.0], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array of angles
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.radians_2"] = radians_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.radians_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.radians_2'.")


check_valid('jax.numpy.radians', generated_inputs['jax.numpy.radians_2'], lib="jax", suffix=2)
