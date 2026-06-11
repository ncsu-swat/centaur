
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sort_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array
    operand = np.array([3.0, 1.0, 2.0, -5.0, 0.0], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "dimension": -1,
        "is_stable": True,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, sort along axis 0
    operand = np.array([[5, 3, 2], [1, 4, 6]], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "dimension": 0,
        "is_stable": False,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, sort along axis 1
    operand = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "operand": operand,
        "dimension": 1,
        "is_stable": True,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with negative values
    operand = np.array([-10, -5, -20, 0, 15, -3], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "dimension": 0,
        "is_stable": True,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array
    operand = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "operand": operand,
        "dimension": -1,
        "is_stable": True,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D int64 array, sort along axis 2
    operand = np.random.randint(-100, 100, size=(2, 2, 3, 2)).astype(np.int64)
    input_dict = {
        "operand": operand,
        "dimension": 2,
        "is_stable": False,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array with NaNs and Infs
    operand = np.array([np.nan, 3.0, -np.inf, np.inf, 2.0, np.nan], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "dimension": -1,
        "is_stable": True,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array with -0.0 and 0.0
    operand = np.array([[-0.0, 0.0, -1.0], [2.0, -0.0, 1.0]], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "dimension": 1,
        "is_stable": True,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int32 array, sort along axis -1
    operand = np.random.randint(-50, 50, size=(3, 3, 3)).astype(np.int32)
    input_dict = {
        "operand": operand,
        "dimension": -1,
        "is_stable": True,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int32 array
    operand = np.array([10, 5, 20, 0, 15], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "dimension": 0,
        "is_stable": False,
        "num_keys": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.sort_1"] = sort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sort_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sort_1'.")


check_valid('jax.lax.sort', generated_inputs['jax.lax.sort_1'], lib="jax", suffix=1)
