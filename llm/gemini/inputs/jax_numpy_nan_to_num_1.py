
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nan_to_num_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, standard replacement
    x = np.array([1.0, np.nan, 2.0, np.inf, -np.inf], dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 0.0,
        "posinf": 999.0,
        "neginf": -999.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float64, check no-copy behavior with large substitutions
    x = np.array([[np.nan, 2.5], [np.inf, -1.5]], dtype=np.float64)
    input_dict = {
        "x": x,
        "copy": False,
        "nan": -1.0,
        "posinf": 100000.0,
        "neginf": -100000.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, float32
    x = np.array([[[np.nan, np.inf], [-np.inf, 1.0]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 3.14,
        "posinf": 100.0,
        "neginf": -100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with no nans/infs, float32
    x = np.array([1.0, 2.0, 3.0, -4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": False,
        "nan": 0.0,
        "posinf": 1.0,
        "neginf": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, float64
    x = np.zeros((2, 2, 2, 2), dtype=np.float64)
    x[0, 0, 0, 0] = np.nan
    x[1, 1, 1, 1] = np.inf
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 99.0,
        "posinf": 9999.0,
        "neginf": -9999.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D array (scalar), float32
    x = np.array(np.nan, dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": False,
        "nan": 0.0,
        "posinf": 1.0,
        "neginf": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with only nans, float32
    x = np.full((3, 3), np.nan, dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": True,
        "nan": -999.0,
        "posinf": 0.0,
        "neginf": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with only infs, float64
    x = np.array([np.inf, -np.inf, np.inf], dtype=np.float64)
    input_dict = {
        "x": x,
        "copy": False,
        "nan": 1.23,
        "posinf": 4.56,
        "neginf": -7.89
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with mixed nans and infs, float32
    x = np.array([[[np.nan], [np.inf]], [[-np.inf], [2.0]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 0.0,
        "posinf": 100.0,
        "neginf": -100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, float64
    x = np.array([[1.1, np.nan], [2.2, np.inf], [3.3, -np.inf]], dtype=np.float64)
    input_dict = {
        "x": x,
        "copy": True,
        "nan": 1.1,
        "posinf": 2.2,
        "neginf": -3.3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nan_to_num_1"] = nan_to_num_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nan_to_num_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nan_to_num_1'.")


check_valid('jax.numpy.nan_to_num', generated_inputs['jax.numpy.nan_to_num_1'], lib="jax", suffix=1)
