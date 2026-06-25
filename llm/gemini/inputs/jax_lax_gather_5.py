
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax
import numpy as np
import copy

class SafeGatherDimensionNumbers(jax.lax.GatherDimensionNumbers):
    def __len__(self):
        return 0

def gather_inputs():
    list_of_inputs = []

    # Case 1: Length 1 dimension numbers, simple 2D row-like gather
    input_dict = {
        'operand': np.random.randn(10, 10).astype(np.float32),
        'start_indices': np.array([[1], [3], [5]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        ),
        'slice_sizes': [1, 3],
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Length 1 dimension numbers, simple 2D column-like gather
    input_dict = {
        'operand': np.random.randn(10, 10).astype(np.float32),
        'start_indices': np.array([[1], [2], [3]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(1,),
            start_index_map=(1,)
        ),
        'slice_sizes': [3, 1],
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Length 1, sorted indices, fill mode
    input_dict = {
        'operand': np.random.randn(8, 8).astype(np.float32),
        'start_indices': np.array([[0], [2], [4]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        ),
        'slice_sizes': [1, 2],
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': "fill",
        'fill_value': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Length 1, small dimensions
    input_dict = {
        'operand': np.random.randn(5, 10).astype(np.float32),
        'start_indices': np.array([[2], [4]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        ),
        'slice_sizes': [1, 4],
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Length 1, unique and unsorted indices
    input_dict = {
        'operand': np.random.randn(10, 5).astype(np.float32),
        'start_indices': np.array([[4], [1], [3]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(1,),
            start_index_map=(1,)
        ),
        'slice_sizes': [4, 1],
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Length 2 dimension numbers, 4D operand
    input_dict = {
        'operand': np.random.randn(5, 5, 5, 5).astype(np.float32),
        'start_indices': np.array([[0, 1], [2, 3], [1, 2]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1, 2),
            collapsed_slice_dims=(0, 1),
            start_index_map=(0, 1)
        ),
        'slice_sizes': [1, 1, 3, 3],
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Length 2, sorted & unique indices
    input_dict = {
        'operand': np.random.randn(6, 6, 6, 6).astype(np.float32),
        'start_indices': np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1, 2),
            collapsed_slice_dims=(0, 1),
            start_index_map=(0, 1)
        ),
        'slice_sizes': [1, 1, 2, 2],
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Length 2, different slices
    input_dict = {
        'operand': np.random.randn(8, 8, 8, 8).astype(np.float32),
        'start_indices': np.array([[1, 1], [3, 3]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1, 2),
            collapsed_slice_dims=(0, 1),
            start_index_map=(0, 1)
        ),
        'slice_sizes': [1, 1, 4, 4],
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Length 2, collapsing different dimensions
    input_dict = {
        'operand': np.random.randn(5, 5, 5, 5).astype(np.float32),
        'start_indices': np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1, 2),
            collapsed_slice_dims=(2, 3),
            start_index_map=(2, 3)
        ),
        'slice_sizes': [3, 3, 1, 1],
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Length 2, fill mode with negative value
    input_dict = {
        'operand': np.random.randn(6, 6, 6, 6).astype(np.float32),
        'start_indices': np.array([[1, 1], [2, 2]], dtype=np.int32),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1, 2),
            collapsed_slice_dims=(1, 2),
            start_index_map=(1, 2)
        ),
        'slice_sizes': [2, 1, 1, 2],
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "fill",
        'fill_value': -999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 11: Length 1, float64
    input_dict = {
        'operand': np.random.randn(12, 12).astype(np.float64),
        'start_indices': np.array([[1], [5]], dtype=np.int64),
        'dimension_numbers': SafeGatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        ),
        'slice_sizes': [1, 6],
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
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
