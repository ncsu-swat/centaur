
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D array, basic float32 update
    operand = np.random.randn(10).astype(np.float32)
    update = np.array([1.5], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 3,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, negative index
    operand = np.arange(5).astype(np.int32)
    update = np.array([9], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": -2,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, update along axis 0
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones((1, 4), dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 1,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, update along axis 1 with allow_negative_indices=False
    operand = np.random.randn(3, 5).astype(np.float64)
    update = np.random.randn(3, 1).astype(np.float64)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 4,
        "axis": 1,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, int32, update along axis 1
    operand = np.ones((2, 3, 4), dtype=np.int32)
    update = np.zeros((2, 1, 4), dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 2,
        "axis": 1,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, negative index along axis 2
    operand = np.random.randn(2, 2, 5).astype(np.float32)
    update = np.random.randn(2, 2, 1).astype(np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": -1,
        "axis": 2,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, float32, update along axis 0
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    update = np.random.randn(1, 2, 2, 2).astype(np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 0,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, smaller update size along remaining axis (broadcasting/slice matching)
    operand = np.zeros((5, 5), dtype=np.float32)
    update = np.ones((1, 3), dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 2,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Out of bounds index (to be clipped)
    operand = np.ones((3, 3), dtype=np.float32)
    update = np.zeros((3, 1), dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 10,
        "axis": 1,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean array update
    operand = np.zeros((8,), dtype=bool)
    update = np.ones((1,), dtype=bool)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 5,
        "axis": 0,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_index_in_dim_1"] = dynamic_update_index_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_index_in_dim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_index_in_dim_1'.")


check_valid('jax.lax.dynamic_update_index_in_dim', generated_inputs['jax.lax.dynamic_update_index_in_dim_1'], lib="jax", suffix=1)
