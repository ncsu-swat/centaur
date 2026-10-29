
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_update_slice_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard update at positive index
    operand = np.zeros(6, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with negative start index
    operand = np.arange(10, dtype=np.int32)
    update = np.array([99, 98], dtype=np.int32)
    start_index = np.array(-3, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, update along axis 0, negative indices disallowed
    operand = np.zeros((5, 4), dtype=np.float64)
    update = np.ones((2, 4), dtype=np.float64)
    start_index = np.array(1, dtype=np.int64)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, update along axis 1 with negative index
    operand = np.ones((3, 6), dtype=np.float32)
    update = np.zeros((3, 2), dtype=np.float32)
    start_index = np.array(-2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D complex64 array, update along axis 2
    operand = np.zeros((2, 3, 4), dtype=np.complex64)
    update = np.ones((2, 3, 2), dtype=np.complex64)
    start_index = np.array(0, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 2,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int64 array, update along axis 1, allow_negative_indices=False
    operand = np.ones((4, 6, 2), dtype=np.int64)
    update = np.zeros((4, 3, 2), dtype=np.int64)
    start_index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 1,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D boolean array, update along axis 0
    operand = np.zeros((4, 2, 3, 2), dtype=np.bool_)
    update = np.ones((2, 2, 3, 2), dtype=np.bool_)
    start_index = np.array(1, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D uint8 array with index clamping (index larger than bounds)
    operand = np.zeros(8, dtype=np.uint8)
    update = np.array([10, 20, 30], dtype=np.uint8)
    start_index = np.array(15, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array where update has smaller size in non-updated dimension
    operand = np.zeros((4, 5), dtype=np.float32)
    update = np.ones((2, 3), dtype=np.float32)
    start_index = np.array(1, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, update along axis 3
    operand = np.zeros((2, 2, 2, 6), dtype=np.float32)
    update = np.ones((2, 2, 2, 3), dtype=np.float32)
    start_index = np.array(2, dtype=np.int64)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 3,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int16 array with start_index of 0 and allow_negative_indices=False
    operand = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    update = np.array([10, 20], dtype=np.int16)
    start_index = np.array(0, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": start_index,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.dynamic_update_slice_in_dim_2"] = (
    dynamic_update_slice_in_dim_inputs()
)

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
