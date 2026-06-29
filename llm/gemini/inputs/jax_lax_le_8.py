
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_le_inputs():
    list_of_inputs = []

    # Input 1: 1D array, int32
    input_dict = {
        "x": 0,
        "y": np.array([-1, 0, 1, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, int32
    input_dict = {
        "x": 5,
        "y": np.random.randint(-10, 10, size=(3, 3), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, int32
    input_dict = {
        "x": -3,
        "y": np.random.randint(-5, 5, size=(2, 2, 2), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, int64
    input_dict = {
        "x": 10,
        "y": np.array([5, 10, 15], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, int32
    input_dict = {
        "x": 0,
        "y": np.random.randint(-1, 2, size=(2, 1, 3, 2), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, int64
    input_dict = {
        "x": -100,
        "y": np.random.randint(-200, 0, size=(4, 4), dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D array (scalar), int32
    input_dict = {
        "x": 42,
        "y": np.array(42, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D array, int32
    input_dict = {
        "x": 1,
        "y": np.random.randint(-5, 5, size=(1, 2, 1, 2, 1), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, int32
    input_dict = {
        "x": -10,
        "y": np.array([-20, -10, 0, 10], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, int64
    input_dict = {
        "x": 1000,
        "y": np.random.randint(500, 1500, size=(10, 10), dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.le_8"] = jax_lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_8'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_8'], lib="jax", suffix=8)
