
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def clamp_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive bounds
    x = np.array([-2, 0, 3, 5, 12], dtype=np.int32)
    input_dict = {"min": 0, "x": x, "max": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with negative to positive bounds
    x = np.array([[-15, -5, 2], [4, 12, 25]], dtype=np.int32)
    input_dict = {"min": -10, "x": x, "max": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 array spanning wide range
    x = np.array([-10, -5, 0, 5, 10, 15], dtype=np.int32)
    input_dict = {"min": -2, "x": x, "max": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array
    x = np.random.randint(-20, 20, size=(2, 3, 4), dtype=np.int32)
    input_dict = {"min": -5, "x": x, "max": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D scalar int32 array
    x = np.array(7, dtype=np.int32)
    input_dict = {"min": 0, "x": x, "max": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 array
    x = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32)
    input_dict = {"min": 25, "x": x, "max": 45}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D int32 array
    x = np.random.randint(-50, 50, size=(2, 2, 3, 2), dtype=np.int32)
    input_dict = {"min": -20, "x": x, "max": 20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array where min equals max
    x = np.array([-1, 0, 1, 2], dtype=np.int32)
    input_dict = {"min": 0, "x": x, "max": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int32 array with negative bounds
    x = np.array([-20, -15, -10, -5, 0], dtype=np.int32)
    input_dict = {"min": -15, "x": x, "max": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty 1D int32 array
    x = np.array([], dtype=np.int32)
    input_dict = {"min": 0, "x": x, "max": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D int32 array with large bounds
    x = np.array([[-100, 0], [50, 200]], dtype=np.int32)
    input_dict = {"min": -50, "x": x, "max": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.clamp_3"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.clamp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.clamp_3'.")


check_valid('jax.lax.clamp', generated_inputs['jax.lax.clamp_3'], lib="jax", suffix=3)
