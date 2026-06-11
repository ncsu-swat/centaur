
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def slice_inputs():
    list_of_inputs = []

    # Input 1: 1D array, slicing a portion
    operand = np.arange(10, dtype=np.int32)
    start_indices = (2,)
    limit_indices = (8,)
    strides = (1,)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    # Input 2: 2D array, slice without striding
    operand = np.random.randn(5, 5).astype(np.float32)
    start_indices = (1, 1)
    limit_indices = (4, 4)
    strides = (1, 1)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    # Input 3: 2D array, slice with striding
    operand = np.random.randn(10, 10).astype(np.float32)
    start_indices = (0, 0)
    limit_indices = (10, 10)
    strides = (2, 2)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    # Input 4: 3D array, float64
    operand = np.random.randn(4, 4, 4).astype(np.float64)
    start_indices = (1, 0, 2)
    limit_indices = (3, 4, 4)
    strides = (1, 1, 1)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    # Input 5: 4D array, slicing with strides
    operand = np.zeros((2, 4, 6, 8), dtype=np.float32)
    start_indices = (0, 0, 0, 0)
    limit_indices = (2, 4, 6, 8)
    strides = (1, 2, 2, 4)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    # Input 6: 1D array with stride > 1, bool dtype
    operand = np.ones(20, dtype=np.bool_)
    start_indices = (5,)
    limit_indices = (15,)
    strides = (3,)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    # Input 7: 2D array of complex64
    operand = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(np.complex64)
    start_indices = (2, 2)
    limit_indices = (5, 5)
    strides = (1, 1)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    # Input 8: 5D array, small sizes, int8 dtype
    operand = np.ones((2, 2, 2, 2, 2), dtype=np.int8)
    start_indices = (0, 0, 0, 0, 0)
    limit_indices = (1, 2, 1, 2, 1)
    strides = (1, 1, 1, 1, 1)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    # Input 9: 3D array, strided slicing on one dimension
    operand = np.random.randn(100, 2, 2).astype(np.float32)
    start_indices = (10, 0, 0)
    limit_indices = (90, 2, 2)
    strides = (10, 1, 1)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    # Input 10: 2D array, slicing to a single element
    operand = np.arange(100).reshape(10, 10).astype(np.int32)
    start_indices = (5, 5)
    limit_indices = (6, 6)
    strides = (1, 1)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': strides
    })

    return list_of_inputs

generated_inputs["jax.lax.slice_2"] = slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.slice_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.slice_2'.")


check_valid('jax.lax.slice', generated_inputs['jax.lax.slice_2'], lib="jax", suffix=2)
