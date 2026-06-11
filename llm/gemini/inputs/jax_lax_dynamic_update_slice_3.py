
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Case 1: Simple 1D float32 array
    operand = np.zeros(10, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_indices = np.array([2], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 1D array with negative start index
    operand = np.zeros(5, dtype=np.float32)
    update = np.ones(2, dtype=np.float32)
    start_indices = np.array([-2], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D array, float32, simple update
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones((2, 2), dtype=np.float32)
    start_indices = np.array([1, 1], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D array with clamping behavior (out-of-bounds start)
    operand = np.zeros((3, 3), dtype=np.float32)
    update = np.ones((2, 2), dtype=np.float32)
    start_indices = np.array([2, 2], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D array, int32 type
    operand = np.zeros((5, 5, 5), dtype=np.int32)
    update = np.ones((2, 2, 2), dtype=np.int32)
    start_indices = np.array([1, 2, 0], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 1D array, float64, start_indices using int64
    operand = np.arange(8).astype(np.float64)
    update = np.ones(4, dtype=np.float64)
    start_indices = np.array([3], dtype=np.int64)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 2D array, int32, allow_negative_indices is False
    operand = np.zeros((6, 6), dtype=np.int32)
    update = np.ones((3, 3), dtype=np.int32)
    start_indices = np.array([0, 3], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Large 1D array
    operand = np.zeros(100, dtype=np.float32)
    update = np.ones(10, dtype=np.float32)
    start_indices = np.array([95], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Boolean types
    operand = np.zeros((4, 4), dtype=bool)
    update = np.ones((2, 2), dtype=bool)
    start_indices = np.array([2, 2], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 4D array
    operand = np.zeros((3, 3, 3, 3), dtype=np.float32)
    update = np.ones((1, 2, 1, 2), dtype=np.float32)
    start_indices = np.array([1, 0, 2, 1], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_3"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_3'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_3'], lib="jax", suffix=3)
