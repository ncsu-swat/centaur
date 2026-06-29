
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_slice_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D array, slicing axis 0, positive start_index
    operand = np.arange(10).astype(np.float32)
    start_index = np.array(2, dtype=np.int32)
    slice_size = 4
    axis = 0
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 2: 2D array, slicing axis 1
    operand = np.random.randn(5, 5).astype(np.float32)
    start_index = np.array(1, dtype=np.int32)
    slice_size = 3
    axis = 1
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 3: 2D array, negative index along axis 0
    operand = np.random.randn(10, 10).astype(np.float64)
    start_index = np.array(-3, dtype=np.int32)
    slice_size = 2
    axis = 0
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 4: 3D array, slicing axis 2 with int64 start index
    operand = np.random.randint(0, 100, size=(4, 4, 4)).astype(np.int32)
    start_index = np.array(0, dtype=np.int64)
    slice_size = 2
    axis = 2
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 5: Large start index that causes clipping
    operand = np.arange(8).astype(np.int64)
    start_index = np.array(10, dtype=np.int32)
    slice_size = 3
    axis = 0
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 6: 4D array, slicing axis 3
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    start_index = np.array(2, dtype=np.int32)
    slice_size = 2
    axis = 3
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 7: 1D array of int16, negative start index
    operand = np.array([10, 20, 30, 40, 50, 60]).astype(np.int16)
    start_index = np.array(-4, dtype=np.int32)
    slice_size = 3
    axis = 0
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 8: 3D float32, slicing axis 1
    operand = np.random.randn(3, 8, 3).astype(np.float32)
    start_index = np.array(5, dtype=np.int32)
    slice_size = 3
    axis = 1
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 9: 2D complex64, slicing axis 0
    operand = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(np.complex64)
    start_index = np.array(2, dtype=np.int32)
    slice_size = 4
    axis = 0
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 10: 5D array, slicing axis 4
    operand = np.zeros((2, 2, 2, 2, 6)).astype(np.float32)
    start_index = np.array(-2, dtype=np.int32)
    slice_size = 2
    axis = 4
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_index': start_index,
        'slice_size': slice_size,
        'axis': axis,
        'allow_negative_indices': allow_negative_indices
    })

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
