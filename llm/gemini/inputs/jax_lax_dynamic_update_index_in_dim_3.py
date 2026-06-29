
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_dynamic_update_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case, float32, positive index
    input_dict = {
        "operand": np.zeros(6, dtype=np.float32),
        "update": 1.0,
        "index": 2,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 operand, negative index
    input_dict = {
        "operand": np.ones(10, dtype=np.float32),
        "update": -3.14,
        "index": -2,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Out of bounds positive index (will be clipped)
    input_dict = {
        "operand": np.arange(5, dtype=np.float32),
        "update": 9.9,
        "index": 10,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Out of bounds negative index
    input_dict = {
        "operand": np.zeros(8, dtype=np.float32),
        "update": 5.5,
        "index": -15,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 operand, allow_negative_indices=False
    input_dict = {
        "operand": np.ones(4, dtype=np.float32),
        "update": 0.0,
        "index": 1,
        "axis": 0,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 operand, another positive index
    input_dict = {
        "operand": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "update": 42.0,
        "index": 3,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large 1D array
    input_dict = {
        "operand": np.zeros(1000, dtype=np.float32),
        "update": -100.5,
        "index": 500,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small 1D array, index 0
    input_dict = {
        "operand": np.array([0.5, 1.5], dtype=np.float32),
        "update": 99.9,
        "index": 0,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 operand, allow_negative_indices=False, different index
    input_dict = {
        "operand": np.ones(7, dtype=np.float32),
        "update": 12.0,
        "index": 4,
        "axis": 0,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative float update, index at the last element
    input_dict = {
        "operand": np.zeros(12, dtype=np.float32),
        "update": -0.0001,
        "index": -1,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_index_in_dim_3"] = jax_lax_dynamic_update_index_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_index_in_dim_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_index_in_dim_3'.")


check_valid('jax.lax.dynamic_update_index_in_dim', generated_inputs['jax.lax.dynamic_update_index_in_dim_3'], lib="jax", suffix=3)
