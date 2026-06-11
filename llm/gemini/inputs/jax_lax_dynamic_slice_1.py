
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_slice_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D slice
    operand = np.arange(10).astype(np.float32)
    start_indices = (2,)
    slice_sizes = (4,)
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 2: 2D array slice
    operand = np.arange(12).reshape(3, 4).astype(np.int32)
    start_indices = (1, 1)
    slice_sizes = (2, 2)
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 3: 2D array with negative start index
    operand = np.arange(20).reshape(4, 5).astype(np.float64)
    start_indices = (-2, 1)
    slice_sizes = (2, 3)
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 4: 3D array slice, no negative indices allowed
    operand = np.random.randn(3, 4, 5).astype(np.float32)
    start_indices = (1, 2, 0)
    slice_sizes = (2, 1, 3)
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 5: 4D array slice
    operand = np.random.randint(0, 10, size=(2, 3, 4, 5)).astype(np.int64)
    start_indices = (0, 1, 2, 1)
    slice_sizes = (1, 2, 2, 3)
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 6: 1D array with slice sizes equal to input size
    operand = np.linspace(0, 1, 100).astype(np.float32)
    start_indices = (0,)
    slice_sizes = (100,)
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 7: 2D array, start index out of bounds (which will be clamped dynamically)
    operand = np.arange(12).reshape(3, 4).astype(np.float32)
    start_indices = (2, 2)
    slice_sizes = (2, 3)
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 8: Complex 3D array
    operand = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex64)
    start_indices = (0, 0, 1)
    slice_sizes = (2, 1, 1)
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 9: 5D array with small dimensions
    operand = np.ones((2, 2, 2, 2, 2), dtype=np.bool_)
    start_indices = (1, 0, 1, 0, 1)
    slice_sizes = (1, 1, 1, 1, 1)
    allow_negative_indices = False
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 10: 2D array of zeros
    operand = np.zeros((5, 5)).astype(np.float32)
    start_indices = (2, 2)
    slice_sizes = (1, 3)
    allow_negative_indices = True
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'slice_sizes': slice_sizes,
        'allow_negative_indices': allow_negative_indices
    })

    return list_of_inputs

generated_inputs["jax.lax.dynamic_slice_1"] = dynamic_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_slice_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_slice_1'.")


check_valid('jax.lax.dynamic_slice', generated_inputs['jax.lax.dynamic_slice_1'], lib="jax", suffix=1)
