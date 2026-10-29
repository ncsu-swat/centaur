
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_slice_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D array, basic slice
    operand = np.arange(10, dtype=np.float32)
    start_index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 4,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, slice along axis 0
    operand = np.random.randn(8, 6).astype(np.float64)
    start_index = np.array(3, dtype=np.int64)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 3,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, slice along axis 1 with negative index
    operand = np.arange(24, dtype=np.int32).reshape(4, 6)
    start_index = np.array(-3, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 2,
        "axis": 1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, slice along axis 2
    operand = np.random.randn(3, 4, 5).astype(np.float32)
    start_index = np.array(1, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 3,
        "axis": 2,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, negative axis index
    operand = np.ones((5, 7, 9), dtype=np.complex64)
    start_index = np.array(2, dtype=np.int64)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 4,
        "axis": -2,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, slice_size 0 (empty slice)
    operand = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    start_index = np.array(1, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 0,
        "axis": 3,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean tensor, slice full dimension length
    operand = np.array([[True, False], [False, True]], dtype=np.bool_)
    start_index = np.array(0, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 2,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with start_index out of bounds (clipped by lax)
    operand = np.linspace(0, 1, 20, dtype=np.float32)
    start_index = np.array(25, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 5,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, slice middle axis
    operand = np.arange(2 * 3 * 4 * 2 * 2, dtype=np.int64).reshape(2, 3, 4, 2, 2)
    start_index = np.array(2, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 2,
        "axis": 2,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, slice size 1 along axis 0
    operand = np.random.uniform(-5, 5, size=(10, 10)).astype(np.float32)
    start_index = np.array(-5, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_index": start_index,
        "slice_size": 1,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.dynamic_slice_in_dim_2"] = dynamic_slice_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_slice_in_dim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_slice_in_dim_2'.")


check_valid('jax.lax.dynamic_slice_in_dim', generated_inputs['jax.lax.dynamic_slice_in_dim_2'], lib="jax", suffix=2)
