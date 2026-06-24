
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
from jax.lax import GatherDimensionNumbers

def gather_inputs():
    list_of_inputs = []

    # Input 1: float32, mode "clip"
    operand = np.random.randn(2, 5, 5).astype(np.float32)
    start_indices = np.array([[[0], [2], [4]], [[1], [3], [0]]]).astype(np.int32)
    dimension_numbers = GatherDimensionNumbers(
        (1,),  # offset_dims
        (2,),  # collapsed_slice_dims
        (0,),  # operand_batch_dims
        (0,),  # start_indices_batch_dims
        (2,)   # start_index_map
    )
    slice_sizes = [2, 5, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, mode "fill"
    operand = np.random.randn(2, 5, 5).astype(np.float64)
    start_indices = np.array([[[1], [1]], [[3], [2]]]).astype(np.int32)
    dimension_numbers = GatherDimensionNumbers(
        (1,),
        (2,),
        (0,),
        (0,),
        (2,)
    )
    slice_sizes = [2, 5, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, mode "drop"
    operand = np.arange(50).reshape(2, 5, 5).astype(np.int32)
    start_indices = np.array([[[0], [3], [1]], [[2], [4], [0]]]).astype(np.int32)
    dimension_numbers = GatherDimensionNumbers(
        (1,),
        (2,),
        (0,),
        (0,),
        (2,)
    )
    slice_sizes = [2, 5, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "drop",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, mode "promise_in_bounds"
    operand = np.arange(50).reshape(2, 5, 5).astype(np.int64)
    start_indices = np.array([[[1], [2]], [[0], [4]]]).astype(np.int64)
    dimension_numbers = GatherDimensionNumbers(
        (1,),
        (2,),
        (0,),
        (0,),
        (2,)
    )
    slice_sizes = [2, 5, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "promise_in_bounds",
        "fill_value": 42
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, mode "clip" with custom fill_value
    operand = np.random.randn(2, 4, 4).astype(np.float32)
    start_indices = np.array([[[0], [2]], [[1], [3]]]).astype(np.int32)
    dimension_numbers = GatherDimensionNumbers(
        (1,),
        (2,),
        (0,),
        (0,),
        (2,)
    )
    slice_sizes = [2, 4, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": -99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, mode "fill" with unsorted indices
    operand = np.random.randn(2, 4, 4).astype(np.float64)
    start_indices = np.array([[[2], [1]], [[3], [0]]]).astype(np.int64)
    dimension_numbers = GatherDimensionNumbers(
        (1,),
        (2,),
        (0,),
        (0,),
        (2,)
    )
    slice_sizes = [2, 4, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, mode "drop" with unsorted indices
    operand = np.random.randn(2, 6, 6).astype(np.float32)
    start_indices = np.array([[[4], [2], [1]], [[5], [0], [3]]]).astype(np.int32)
    dimension_numbers = GatherDimensionNumbers(
        (1,),
        (2,),
        (0,),
        (0,),
        (2,)
    )
    slice_sizes = [2, 6, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "drop",
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32, mode "promise_in_bounds" with unsorted indices
    operand = np.arange(72).reshape(2, 6, 6).astype(np.int32)
    start_indices = np.array([[[3], [1]], [[2], [4]]]).astype(np.int32)
    dimension_numbers = GatherDimensionNumbers(
        (1,),
        (2,),
        (0,),
        (0,),
        (2,)
    )
    slice_sizes = [2, 6, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, large operand, mode "clip"
    operand = np.random.randn(2, 8, 8).astype(np.float32)
    start_indices = np.array([[[7], [5], [3]], [[1], [4], [6]]]).astype(np.int32)
    dimension_numbers = GatherDimensionNumbers(
        (1,),
        (2,),
        (0,),
        (0,),
        (2,)
    )
    slice_sizes = [2, 8, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64, large operand, mode "fill"
    operand = np.arange(128).reshape(2, 8, 8).astype(np.int64)
    start_indices = np.array([[[0], [4]], [[7], [3]]]).astype(np.int64)
    dimension_numbers = GatherDimensionNumbers(
        (1,),
        (2,),
        (0,),
        (0,),
        (2,)
    )
    slice_sizes = [2, 8, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": -2
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
