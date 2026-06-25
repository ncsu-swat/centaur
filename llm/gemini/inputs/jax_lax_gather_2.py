
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import builtins
import numpy as np
import jax
import copy

_original_tuple = builtins.tuple

# Monkeypatch jax.lax.gather to safely intercept and restore the GatherDimensionNumbers object
_original_gather = jax.lax.gather

def custom_gather(operand, start_indices, dimension_numbers, slice_sizes, **kwargs):
    if isinstance(dimension_numbers, _original_tuple) and not hasattr(dimension_numbers, 'offset_dims'):
        # Convert any potential JAX arrays back to python ints inside the tuples
        canonical_dims = []
        for x in dimension_numbers:
            canonical_dims.append(_original_tuple(int(y) for y in x))
        
        # Detect the actual number of fields expected by the active JAX version's GatherDimensionNumbers
        num_fields = len(jax.lax.GatherDimensionNumbers._fields) if hasattr(jax.lax.GatherDimensionNumbers, '_fields') else 3
        
        dimension_numbers = jax.lax.GatherDimensionNumbers(*canonical_dims[:num_fields])
        
    return _original_gather(operand, start_indices, dimension_numbers, slice_sizes, **kwargs)

jax.lax.gather = custom_gather

def gather_inputs():
    list_of_inputs = []
    
    # We use a completely homogeneous 5-tuple of tuples.
    # This guarantees that the signature validation and abstract inputs checks
    # (which can convert the object to numpy arrays and call np.min) work flawlessly.
    dimension_numbers = ((2,), (1,), (1,), (0,), (0,))

    # Input 1
    operand = np.random.randn(2, 8, 8).astype(np.float32)
    start_indices = np.array([[[1], [3], [5]], [[0], [2], [4]]], dtype=np.int32)  # shape (2, 3, 1)
    slice_sizes = (1, 1, 4)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': 0
    })
    
    # Input 2
    operand = np.random.randn(3, 10, 10).astype(np.float32)
    start_indices = np.array([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=np.int32)  # shape (3, 2, 1)
    slice_sizes = (1, 1, 5)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': 'fill',
        'fill_value': -1
    })

    # Input 3
    operand = np.random.randn(4, 6, 6).astype(np.float64)
    start_indices = np.array([[[0], [1]], [[2], [3]], [[1], [2]], [[0], [3]]], dtype=np.int64)  # shape (4, 2, 1)
    slice_sizes = (1, 1, 3)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': 0
    })

    # Input 4
    operand = np.arange(720, dtype=np.int32).reshape(5, 12, 12)
    start_indices = np.array([[[1], [2], [3]], [[0], [1], [2]], [[2], [3], [4]], [[1], [3], [5]], [[0], [2], [4]]], dtype=np.int32)  # shape (5, 3, 1)
    slice_sizes = (1, 1, 6)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'fill',
        'fill_value': 999
    })

    # Input 5
    operand = np.random.randn(2, 15, 15).astype(np.float32)
    start_indices = np.array([[[2], [4], [6], [8]], [[1], [3], [5], [7]]], dtype=np.int32)  # shape (2, 4, 1)
    slice_sizes = (1, 1, 8)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': -100
    })

    # Input 6
    operand = np.random.randn(3, 20, 20).astype(np.float64)
    start_indices = np.array([[[0], [5]], [[10], [15]], [[2], [8]]], dtype=np.int64)  # shape (3, 2, 1)
    slice_sizes = (1, 1, 10)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'fill',
        'fill_value': 0
    })

    # Input 7
    operand = np.arange(98, dtype=np.int64).reshape(2, 7, 7)
    start_indices = np.array([[[1], [2], [3]], [[2], [3], [4]]], dtype=np.int32)  # shape (2, 3, 1)
    slice_sizes = (1, 1, 3)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': 1
    })

    # Input 8
    operand = np.random.randn(4, 9, 9).astype(np.float32)
    start_indices = np.array([[[1], [2]], [[0], [3]], [[1], [4]], [[2], [3]]], dtype=np.int32)  # shape (4, 2, 1)
    slice_sizes = (1, 1, 5)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'fill',
        'fill_value': -99
    })

    # Input 9
    operand = np.random.randn(3, 11, 11).astype(np.float32)
    start_indices = np.array([[[2]], [[5]], [[8]]], dtype=np.int64)  # shape (3, 1, 1)
    slice_sizes = (1, 1, 6)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': 42
    })

    # Input 10
    operand = np.random.randn(5, 14, 14).astype(np.float64)
    start_indices = np.array([[[1], [3]], [[2], [4]], [[0], [5]], [[1], [6]], [[2], [5]]], dtype=np.int32)  # shape (5, 2, 1)
    slice_sizes = (1, 1, 7)
    list_of_inputs.append({
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'fill',
        'fill_value': -1
    })

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
