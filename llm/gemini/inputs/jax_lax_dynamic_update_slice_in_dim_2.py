
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_dynamic_update_slice_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D update
    operand = np.zeros(10, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_index = np.array(2, dtype=np.int32)
    axis = 0
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D update with negative start index
    operand = np.zeros(10, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_index = np.array(-4, dtype=np.int32)
    axis = 0
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, updating along axis 0 (rows)
    operand = np.zeros((5, 5), dtype=np.float32)
    update = np.ones((2, 5), dtype=np.float32)
    start_index = np.array(1, dtype=np.int32)
    axis = 0
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, updating along axis 1 (columns)
    operand = np.zeros((4, 6), dtype=np.float32)
    update = np.ones((4, 2), dtype=np.float32)
    start_index = np.array(3, dtype=np.int32)
    axis = 1
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array update
    operand = np.zeros((2, 4, 3), dtype=np.float32)
    update = np.ones((2, 2, 3), dtype=np.float32)
    start_index = np.array(1, dtype=np.int32)
    axis = 1
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 type and int64 index
    operand = np.zeros((3, 3, 5), dtype=np.float64)
    update = np.ones((3, 3, 2), dtype=np.float64)
    start_index = np.array(2, dtype=np.int64)
    axis = 2
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer array input
    operand = np.zeros((8,), dtype=np.int32)
    update = np.array([10, 11, 12], dtype=np.int32)
    start_index = np.array(4, dtype=np.int32)
    axis = 0
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boolean array input
    operand = np.zeros((3, 3), dtype=bool)
    update = np.ones((3, 1), dtype=bool)
    start_index = np.array(1, dtype=np.int32)
    axis = 1
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex array input
    operand = np.zeros((4, 4), dtype=np.complex64)
    update = np.ones((1, 4), dtype=np.complex64) * (1 + 1j)
    start_index = np.array(2, dtype=np.int32)
    axis = 0
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Index overflow (will clamp/fit)
    operand = np.zeros((5,), dtype=np.float32)
    update = np.ones((3,), dtype=np.float32)
    start_index = np.array(10, dtype=np.int32)
    axis = 0
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": axis,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_in_dim_2"] = jax_lax_dynamic_update_slice_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_in_dim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_in_dim_2'.")


check_valid('jax.lax.dynamic_update_slice_in_dim', generated_inputs['jax.lax.dynamic_update_slice_in_dim_2'], lib="jax", suffix=2)
