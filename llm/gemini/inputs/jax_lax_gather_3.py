
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Custom GatherDimensionNumbers that inherits from tuple to satisfy both JAX
# attribute requirements and the testing framework's tuple homogeneity checks.
class GatherDimensionNumbers(tuple):
    __slots__ = ()
    @property
    def offset_dims(self): return self[0]
    @property
    def collapsed_slice_dims(self): return self[1]
    @property
    def start_index_map(self): return self[2]

def gather_inputs():
    list_of_inputs = []

    # Homogeneous GatherDimensionNumbers with 3 fields, each of length 1
    dimension_numbers = GatherDimensionNumbers(((1,), (0,), (0,)))

    # Input 1: operand (4, 4), start_indices (3, 1), slice_sizes (1, 4)
    operand = np.random.randn(4, 4).astype(np.float32)
    start_indices = np.array([[0], [2], [3]], dtype=np.int32)
    slice_sizes = (1, 4)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: operand (6, 6), start_indices (4, 1), slice_sizes (1, 6)
    operand = np.random.randn(6, 6).astype(np.float64)
    start_indices = np.array([[0], [1], [2], [3]], dtype=np.int32)
    slice_sizes = (1, 6)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': "promise_in_bounds",
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: operand (3, 8), start_indices (2, 1), slice_sizes (1, 8)
    operand = np.random.randint(-10, 10, size=(3, 8)).astype(np.int32)
    start_indices = np.array([[0], [2]], dtype=np.int32)
    slice_sizes = (1, 8)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': "fill",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: operand (8, 4), start_indices (4, 1), slice_sizes (1, 4)
    operand = np.random.randn(8, 4).astype(np.float32)
    start_indices = np.array([[7], [6], [3], [0]], dtype=np.int32)
    slice_sizes = (1, 4)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: operand (8, 12), start_indices (4, 1), slice_sizes (1, 12)
    operand = np.random.randn(8, 12).astype(np.float32)
    start_indices = np.array([[1], [3], [6], [7]], dtype=np.int32)
    slice_sizes = (1, 12)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "promise_in_bounds",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: operand (12, 12), start_indices (1, 1), slice_sizes (1, 12)
    operand = np.random.randn(12, 12).astype(np.float64)
    start_indices = np.array([[0]], dtype=np.int32)
    slice_sizes = (1, 12)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': "fill",
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: operand (7, 3), start_indices (6, 1), slice_sizes (1, 3)
    operand = np.random.randint(-100, 100, size=(7, 3)).astype(np.int64)
    start_indices = np.array([[0], [1], [2], [3], [4], [6]], dtype=np.int32)
    slice_sizes = (1, 3)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': "clip",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: operand (12, 6), start_indices (4, 1), slice_sizes (1, 6)
    operand = np.random.randn(12, 6).astype(np.float32)
    start_indices = np.array([[11], [8], [6], [4]], dtype=np.int32)
    slice_sizes = (1, 6)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': False,
        'mode': "promise_in_bounds",
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: operand (9, 9), start_indices (3, 1), slice_sizes (1, 9)
    operand = np.random.randint(0, 10, size=(9, 9)).astype(np.int32)
    start_indices = np.array([[1], [2], [3]], dtype=np.int32)
    slice_sizes = (1, 9)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': "fill",
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: operand (10, 2), start_indices (4, 1), slice_sizes (1, 2)
    operand = np.random.randn(10, 2).astype(np.float32)
    start_indices = np.array([[0], [2], [4], [6]], dtype=np.int32)
    slice_sizes = (1, 2)
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': "clip",
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
