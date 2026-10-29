
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def dynamic_update_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, middle index
    input_dict = {
        "operand": np.zeros(6, dtype=np.float32),
        "update": 1.0,
        "index": np.array(2, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Start index 0
    input_dict = {
        "operand": np.ones(5, dtype=np.float32),
        "update": 0.0,
        "index": np.array(0, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative index with allow_negative_indices=True
    input_dict = {
        "operand": np.arange(8, dtype=np.float32),
        "update": -5.5,
        "index": np.array(-1, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative axis (-1)
    input_dict = {
        "operand": np.zeros(4, dtype=np.float32),
        "update": 3.14,
        "index": np.array(1, dtype=np.int64),
        "axis": -1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single-element array
    input_dict = {
        "operand": np.array([10.0], dtype=np.float32),
        "update": 20.0,
        "index": np.array(0, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Out of bounds positive index (clipped)
    input_dict = {
        "operand": np.zeros(6, dtype=np.float32),
        "update": 9.9,
        "index": np.array(10, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative index with axis=-1
    input_dict = {
        "operand": np.full(10, 2.0, dtype=np.float32),
        "update": -1.0,
        "index": np.array(-3, dtype=np.int64),
        "axis": -1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 1D array with allow_negative_indices=False
    input_dict = {
        "operand": np.linspace(0.0, 1.0, 50, dtype=np.float32),
        "update": 42.0,
        "index": np.array(25, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative index with allow_negative_indices=False
    input_dict = {
        "operand": np.zeros(7, dtype=np.float32),
        "update": 7.7,
        "index": np.array(-1, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative float update and int32 index
    input_dict = {
        "operand": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "update": -100.25,
        "index": np.array(2, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_index_in_dim_4"] = dynamic_update_index_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_index_in_dim_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_index_in_dim_4'.")


check_valid('jax.lax.dynamic_update_index_in_dim', generated_inputs['jax.lax.dynamic_update_index_in_dim_4'], lib="jax", suffix=4)
