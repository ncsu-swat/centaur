
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

def gather_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 gather (homogeneous dimension numbers of length 1)
    operand = np.random.randn(10, 5).astype(np.float32)
    start_indices = np.array([[1], [3], [0]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    slice_sizes = [1, 5]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 gather (homogeneous dimension numbers of length 1)
    operand = np.random.randn(12, 6).astype(np.float64)
    start_indices = np.array([[4], [1], [11], [2]], dtype=np.int64)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    slice_sizes = [1, 6]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "fill",
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 gather (homogeneous dimension numbers of length 1)
    operand = np.random.randint(-50, 50, size=(8, 8)).astype(np.int32)
    start_indices = np.array([[2], [5]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    slice_sizes = [1, 8]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "promise_in_bounds",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D gather on axis 1 (homogeneous dimension numbers of length 1)
    operand = np.random.randn(5, 10).astype(np.float32)
    start_indices = np.array([[2], [4]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(1,), start_index_map=(1,)
    )
    slice_sizes = [5, 1]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D Boolean gather (homogeneous dimension numbers of length 1)
    operand = np.random.choice([True, False], size=(6, 6)).astype(np.bool_)
    start_indices = np.array([[0], [4], [3]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    slice_sizes = [1, 6]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 gather, sorted indices (homogeneous dimension numbers of length 1)
    operand = np.random.randn(15, 3).astype(np.float32)
    start_indices = np.array([[1], [5], [9]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    slice_sizes = [1, 3]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D Complex64 gather (homogeneous dimension numbers of length 1)
    operand = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    start_indices = np.array([[0], [2]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    slice_sizes = [1, 4]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int64 gather (homogeneous dimension numbers of length 1)
    operand = np.random.randint(-100, 100, size=(20, 2)).astype(np.int64)
    start_indices = np.array([[1], [15], [8]], dtype=np.int64)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    slice_sizes = [1, 2]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": False,
        "mode": "clip",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 gather (homogeneous dimension numbers of length 2)
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    start_indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1, 2), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)
    )
    slice_sizes = [1, 1, 4, 5]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": True,
        "indices_are_sorted": True,
        "mode": "clip",
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float64 gather (homogeneous dimension numbers of length 2)
    operand = np.random.randn(4, 4, 2, 3).astype(np.float64)
    start_indices = np.array([[0, 2], [1, 3]], dtype=np.int64)
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1, 2), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)
    )
    slice_sizes = [1, 1, 2, 3]
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "dimension_numbers": dimension_numbers,
        "slice_sizes": slice_sizes,
        "unique_indices": False,
        "indices_are_sorted": False,
        "mode": "fill",
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_6"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_6'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_6'], lib="jax", suffix=6)
