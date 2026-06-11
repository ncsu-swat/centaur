
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_slice_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D slice
    operand = np.arange(10, dtype=np.int32)
    start_indices = [2]
    slice_sizes = [4]
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 2: 2D submatrix slice
    operand = np.arange(12, dtype=np.float32).reshape(3, 4)
    start_indices = [1, 1]
    slice_sizes = [2, 2]
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 3: 3D slice with negative indices allowed
    operand = np.random.randn(4, 4, 4).astype(np.float32)
    start_indices = [-3, 1, -2]
    slice_sizes = [2, 2, 2]
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 4: 4D slice with float64 data type
    operand = np.zeros((2, 3, 4, 5), dtype=np.float64)
    start_indices = [0, 1, 2, 3]
    slice_sizes = [2, 1, 2, 1]
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 5: 1D slice with negative index
    operand = np.arange(8, dtype=np.float32)
    start_indices = [-3]
    slice_sizes = [2]
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 6: 2D slice with out-of-bounds start, clamping expected
    operand = np.arange(15, dtype=np.int32).reshape(5, 3)
    start_indices = [4, 2]
    slice_sizes = [2, 2]
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 7: Full size slice of a 2D array
    operand = np.arange(4, dtype=np.float32).reshape(2, 2)
    start_indices = [0, 0]
    slice_sizes = [2, 2]
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 8: High dimensional slice
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    start_indices = [0, 1, 0, 1, 0]
    slice_sizes = [1, 1, 2, 1, 2]
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 9: 2D slice with int16 data type
    operand = np.arange(20, dtype=np.int16).reshape(4, 5)
    start_indices = [-1, -2]
    slice_sizes = [1, 2]
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 10: 1D slice, large array, allow_negative_indices is False
    operand = np.arange(1000, dtype=np.int64)
    start_indices = [500]
    slice_sizes = [100]
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 11: 3D slice, uint8 data type
    operand = np.ones((10, 10, 10), dtype=np.uint8)
    start_indices = [2, 3, 4]
    slice_sizes = [5, 5, 5]
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    return list_of_inputs

generated_inputs["jax.lax.dynamic_slice_2"] = dynamic_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_slice_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_slice_2'.")


check_valid('jax.lax.dynamic_slice', generated_inputs['jax.lax.dynamic_slice_2'], lib="jax", suffix=2)
