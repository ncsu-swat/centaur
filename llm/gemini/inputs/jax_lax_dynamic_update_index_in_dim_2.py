
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def dynamic_update_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, 0D scalar update, axis 0
    operand = np.zeros((6,), dtype=np.float32)
    update = np.array(1.5, dtype=np.float32)
    index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array, 1D update with size 1, negative index
    operand = np.ones((5,), dtype=np.float32)
    update = np.array([42.0], dtype=np.float32)
    index = np.array(-1, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, 1D update matching dimension along axis 0
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones((4,), dtype=np.float32)
    index = np.array(1, dtype=np.int64)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, 2D update with size 1 along axis 0
    operand = np.zeros((4, 5), dtype=np.float64)
    update = np.ones((1, 3), dtype=np.float64)
    index = np.array(3, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int32 array, update along axis 1
    operand = np.zeros((3, 6), dtype=np.int32)
    update = np.full((3,), 7, dtype=np.int32)
    index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 1,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, 2D update along axis 1
    operand = np.zeros((2, 5, 3), dtype=np.float32)
    update = np.ones((2, 3), dtype=np.float32)
    index = np.array(-2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int64 array, 3D update with shape (2, 1, 4) along axis 1
    operand = np.zeros((2, 3, 4), dtype=np.int64)
    update = np.ones((2, 1, 4), dtype=np.int64)
    index = np.array(0, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 1,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, 3D update along axis 3
    operand = np.zeros((2, 2, 2, 4), dtype=np.float32)
    update = np.ones((2, 2, 2), dtype=np.float32)
    index = np.array(3, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 3,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D bool array, 0D boolean update
    operand = np.array([False, False, False, False], dtype=np.bool_)
    update = np.array(True, dtype=np.bool_)
    index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex64 array, 2D update
    operand = np.zeros((3, 3), dtype=np.complex64)
    update = np.array([[1.0 + 2.0j, 2.0 + 0.0j]], dtype=np.complex64)
    index = np.array(1, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Out-of-bounds index that gets clamped
    operand = np.zeros((5,), dtype=np.float32)
    update = np.array([99.0], dtype=np.float32)
    index = np.array(10, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": index,
        "axis": 0,
        "allow_negative_indices": True,
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
