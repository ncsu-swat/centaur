
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
import copy

# Monkeypatch jax.lax.gather to reconstruct GatherDimensionNumbers from plain tuples
original_gather = jax.lax.gather

def patched_gather(operand, start_indices, dimension_numbers, slice_sizes, **kwargs):
    if isinstance(dimension_numbers, tuple) and not isinstance(dimension_numbers, jax.lax.GatherDimensionNumbers):
        dimension_numbers = jax.lax.GatherDimensionNumbers(*dimension_numbers)
    return original_gather(operand, start_indices, dimension_numbers, slice_sizes, **kwargs)

jax.lax.gather = patched_gather

def gather_inputs():
    list_of_inputs = []

    # Case 1: Standard float32 gather
    operand = np.arange(300).reshape(10, 10, 3).astype(np.float32)
    start_indices = np.array([[[2], [2], [2]], [[5], [5], [5]], [[7], [7], [7]], [[1], [1], [1]], [[0], [0], [0]]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 10, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float64 operand
    operand = np.random.randn(8, 12, 4).astype(np.float64)
    start_indices = np.zeros((3, 4, 1), dtype=np.int64)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 5, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'fill',
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Float32 operand with collapsing slice dimensions
    operand = np.random.randn(15, 5, 2).astype(np.float32)
    start_indices = np.zeros((2, 2, 1), dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (3, 1, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'promise_in_bounds',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Int32 operand with small slice sizes
    operand = np.arange(2000).reshape(20, 20, 5).astype(np.int32)
    start_indices = np.zeros((1, 5, 1), dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 1, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D float32 operand
    operand = np.random.randn(6, 8, 2).astype(np.float32)
    start_indices = np.zeros((4, 2, 1), dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 8, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 3D float64, index vector of int64
    operand = np.random.randn(7, 7, 3).astype(np.float64)
    start_indices = np.zeros((2, 3, 1), dtype=np.int64)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 7, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Larger dimensions, 3D float32
    operand = np.random.randn(12, 12, 2).astype(np.float32)
    start_indices = np.zeros((5, 2, 1), dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 12, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'fill',
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Square slice dimensions
    operand = np.random.randn(9, 9, 4).astype(np.float32)
    start_indices = np.zeros((3, 4, 1), dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 9, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'promise_in_bounds',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Int64 operand, large integers
    operand = np.arange(1210).reshape(11, 11, 10).astype(np.int64)
    start_indices = np.zeros((2, 10, 1), dtype=np.int64)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 11, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 1-sized batch gather
    operand = np.random.randn(14, 14, 2).astype(np.float32)
    start_indices = np.zeros((1, 2, 1), dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(0,),
        operand_batching_dims=(2,),
        start_indices_batching_dims=(1,),
        start_index_map=(0,)
    )
    slice_sizes = (1, 14, 1)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_3"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_3'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_3'], lib="jax", suffix=3)
