
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array, positive index
    input_dict = {
        "operand": np.zeros(6, dtype=np.float32),
        "update": 1.0,
        "index": np.array(2, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, out of bounds positive index (will be clipped)
    input_dict = {
        "operand": np.ones(5, dtype=np.float64),
        "update": 2.5,
        "index": np.array(10, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative index, allow_negative_indices is True
    input_dict = {
        "operand": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        "update": -5.0,
        "index": np.array(-1, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative index, allow_negative_indices is False
    input_dict = {
        "operand": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "update": 0.0,
        "index": np.array(-2, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large 1D array, int64 index
    input_dict = {
        "operand": np.linspace(0.0, 10.0, 100, dtype=np.float32),
        "update": 99.9,
        "index": np.array(50, dtype=np.int64),
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element array
    input_dict = {
        "operand": np.array([0.5], dtype=np.float32),
        "update": 1.5,
        "index": np.array(0, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Axis is -1 (which is equivalent to 0 for 1D)
    input_dict = {
        "operand": np.arange(10, dtype=np.float32),
        "update": 12.34,
        "index": np.array(4, dtype=np.int32),
        "axis": -1,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Index as 0D array (scalar) with float64 operand
    input_dict = {
        "operand": np.zeros(8, dtype=np.float64),
        "update": -1.1,
        "index": np.array(6, dtype=np.int16),
        "axis": 0,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large update value on a smaller array
    input_dict = {
        "operand": np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32),
        "update": 1000.0,
        "index": np.array(3, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Index is out of bounds on the negative side
    input_dict = {
        "operand": np.array([9.0, 8.0, 7.0], dtype=np.float32),
        "update": -9.0,
        "index": np.array(-10, dtype=np.int32),
        "axis": 0,
        "allow_negative_indices": True
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
