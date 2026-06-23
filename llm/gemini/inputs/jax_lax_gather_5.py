
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

class SafeGatherDimensionNumbers(jax.lax.GatherDimensionNumbers):
    def __array__(self, *args, **kwargs):
        return np.array([0, 0, 0, 0, 0])

def gather_inputs():
    list_of_inputs = []

    # Input 1: Rank 2, simple gather
    operand = np.random.randn(8, 8).astype(np.float32)
    start_indices = np.array([[2], [4], [6]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = [1, 2]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rank 2, mapping to dimension 1
    operand = np.random.randn(6, 6).astype(np.float32)
    start_indices = np.array([[1], [3], [2]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(0,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,)
    )
    slice_sizes = [2, 1]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': 'fill',
        'fill_value': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rank 4, gathering dimensions
    operand = np.random.randn(4, 4, 4, 4).astype(np.float32)
    start_indices = np.array([[1, 2], [3, 1]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(2, 3),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    slice_sizes = [1, 1, 2, 2]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rank 4, gathering other dimensions
    operand = np.random.randn(6, 6, 6, 6).astype(np.float32)
    start_indices = np.array([[1, 2], [3, 2], [0, 1]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(0, 1),
        collapsed_slice_dims=(2, 3),
        start_index_map=(2, 3)
    )
    slice_sizes = [2, 2, 1, 1]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'drop',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rank 2, multi-dimensional start indices
    operand = np.random.randn(8, 8).astype(np.float32)
    start_indices = np.array([[[2], [4], [3]], [[1], [4], [2]]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = [1, 3]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rank 4, multi-dimensional start indices
    operand = np.random.randn(4, 4, 4, 4).astype(np.float32)
    start_indices = np.array([[[0, 0], [1, 1]], [[2, 2], [1, 2]]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(2, 3),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    slice_sizes = [1, 1, 2, 2]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'fill',
        'fill_value': -10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rank 2, float64 operand
    operand = np.random.randn(8, 8).astype(np.float64)
    start_indices = np.array([[0], [2], [4], [6]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = [1, 4]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rank 2, int32 operand
    operand = np.random.randint(-50, 50, size=(8, 8)).astype(np.int32)
    start_indices = np.array([[1], [3]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(0,),
        collapsed_slice_dims=(1,),
        start_index_map=(1,)
    )
    slice_sizes = [4, 1]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': 42
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 4, int64 start indices
    operand = np.random.randn(4, 4, 4, 4).astype(np.float32)
    start_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(2, 3),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    slice_sizes = [1, 1, 3, 3]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': 'fill',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rank 2, empty start indices
    operand = np.random.randn(4, 4).astype(np.float32)
    start_indices = np.zeros((0, 1), dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slice_sizes = [1, 2]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_5"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_5'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_5'], lib="jax", suffix=5)
