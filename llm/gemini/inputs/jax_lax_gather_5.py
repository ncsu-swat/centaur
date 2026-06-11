
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
from jax import lax

class SafeGatherDimensionNumbers(lax.GatherDimensionNumbers):
    def __new__(cls, *args, **kwargs):
        if kwargs:
            return super().__new__(cls, *args, **kwargs)
        for i in range(len(args), -1, -1):
            try:
                return super().__new__(cls, *args[:i])
            except TypeError:
                continue
        raise TypeError("Failed to construct GatherDimensionNumbers")

    def __array__(self, *args, **kwargs):
        return np.array([0])

def gather_inputs():
    list_of_inputs = []

    # Input 1: 2D operand, length 1 dimension numbers
    operand = np.arange(20).reshape(4, 5).astype(np.float32)
    start_indices = np.array([[0], [2], [1]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 5],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D operand, sorted unique indices
    operand = np.arange(12).reshape(3, 4).astype(np.int32)
    start_indices = np.array([[1], [2]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 4],
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D operand, fill mode
    operand = np.arange(15).reshape(5, 3).astype(np.float64)
    start_indices = np.array([[4], [0], [2]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 3],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D operand, different fill_value
    operand = np.arange(30).reshape(6, 5).astype(np.float32)
    start_indices = np.array([[5], [1]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 5],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D operand, unique unsorted indices with fill mode
    operand = np.arange(16).reshape(4, 4).astype(np.int32)
    start_indices = np.array([[2], [3], [0]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 4],
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": -999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D operand, length 2 dimension numbers
    operand = np.arange(16).reshape(2, 2, 2, 2).astype(np.float32)
    start_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 1, 2, 2],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D operand, unique indices
    operand = np.arange(81).reshape(3, 3, 3, 3).astype(np.int32)
    start_indices = np.array([[0, 1], [2, 0], [1, 2]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 1, 3, 3],
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D operand, float64 types
    operand = np.arange(24).reshape(2, 3, 2, 2).astype(np.float64)
    start_indices = np.array([[1, 2], [0, 1]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 1, 2, 2],
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": -99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D operand, sorted indices
    operand = np.arange(48).reshape(3, 2, 4, 2).astype(np.float32)
    start_indices = np.array([[0, 0], [1, 1], [2, 1]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 1, 4, 2],
        "unique_indices": False,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D operand, unique and sorted indices
    operand = np.arange(16).reshape(2, 2, 2, 2).astype(np.int32)
    start_indices = np.array([[0, 0], [1, 0]], dtype=np.int32)
    dimension_numbers = SafeGatherDimensionNumbers(
        offset_dims=(1, 2),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": [1, 1, 2, 2],
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": -1
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
