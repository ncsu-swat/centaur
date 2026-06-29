
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array update with 1D update array
    operand = np.zeros(6, dtype=np.float32)
    update = np.array([1.0], dtype=np.float32)
    index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, scalar update, negative index
    operand = np.zeros(6, dtype=np.float32)
    update = np.array(1.0, dtype=np.float32)
    index = np.array(-2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, update along axis 0
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones(4, dtype=np.float32)
    index = np.array(1, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, update along axis 1, negative index disallowed
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones(4, dtype=np.float32)
    index = np.array(3, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 1,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, update along axis 1
    operand = np.zeros((2, 3, 4), dtype=np.float32)
    update = np.ones((2, 4), dtype=np.float32)
    index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 1,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, update along axis 2 with negative index
    operand = np.zeros((2, 3, 4), dtype=np.float32)
    update = np.ones((2, 3), dtype=np.float32)
    index = np.array(-1, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 2,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, float64 type
    operand = np.zeros((2, 2, 2, 2), dtype=np.float64)
    update = np.ones((2, 2, 2), dtype=np.float64)
    index = np.array(0, dtype=np.int64)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 3,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Out of bounds index (to be clipped)
    operand = np.zeros(5, dtype=np.float32)
    update = np.array([5.0], dtype=np.float32)
    index = np.array(10, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Integer array operand
    operand = np.zeros((3, 3), dtype=np.int32)
    update = np.ones(3, dtype=np.int32) * 9
    index = np.array(1, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 1,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, update shape with extra dimension of size 1
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones((1, 4), dtype=np.float32)
    index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_index_in_dim_2"] = dynamic_update_index_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_index_in_dim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_index_in_dim_2'.")


check_valid('jax.lax.dynamic_update_index_in_dim', generated_inputs['jax.lax.dynamic_update_index_in_dim_2'], lib="jax", suffix=2)
