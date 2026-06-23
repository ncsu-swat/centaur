
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.lax

# Monkeypatch jax.lax.gather to seamlessly accept plain tuples for dimension_numbers
_original_gather = jax.lax.gather

def patched_gather(operand, start_indices, dimension_numbers, slice_sizes, *args, **kwargs):
    if isinstance(dimension_numbers, tuple) and not isinstance(dimension_numbers, jax.lax.GatherDimensionNumbers):
        dimension_numbers = jax.lax.GatherDimensionNumbers(*dimension_numbers)
    return _original_gather(operand, start_indices, dimension_numbers, slice_sizes, *args, **kwargs)

jax.lax.gather = patched_gather

def gather_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D gather, size 10x10
    operand = np.random.randn(10, 10).astype(np.float32)
    start_indices = np.array([[2], [4]], dtype=np.int32)
    slice_sizes = (1, 2)
    dimension_numbers = ((1,), (0,), (0,))
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small 5x5 gather with sorted indices
    operand = np.random.randn(5, 5).astype(np.float32)
    start_indices = np.array([[1], [3]], dtype=np.int32)
    slice_sizes = (1, 1)
    dimension_numbers = ((1,), (0,), (0,))
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': "fill",
        'fill_value': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small 8x8 gather, promise_in_bounds mode
    operand = np.random.randn(8, 8).astype(np.float32)
    start_indices = np.array([[0], [2]], dtype=np.int32)
    slice_sizes = (1, 4)
    dimension_numbers = ((1,), (0,), (0,))
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': "promise_in_bounds",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small 6x6 gather, drop mode
    operand = np.random.randn(6, 6).astype(np.float32)
    start_indices = np.array([[1], [2]], dtype=np.int32)
    slice_sizes = (1, 3)
    dimension_numbers = ((1,), (0,), (0,))
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "drop",
        'fill_value': 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small 4x4 gather with custom float bounds
    operand = np.random.uniform(-10, 10, size=(4, 4)).astype(np.float32)
    start_indices = np.array([[0], [1]], dtype=np.int32)
    slice_sizes = (1, 2)
    dimension_numbers = ((0,), (0,), (0,))
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_2"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_2'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_2'], lib="jax", suffix=2)
