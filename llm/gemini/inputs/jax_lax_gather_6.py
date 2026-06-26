
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

def gather_inputs():
    list_of_inputs = []

    # Input 1
    operand = np.random.randn(4, 3).astype(np.float32)
    start_indices = np.array([[1], [3]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
    slice_sizes = [1, 3]
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

    # Input 2
    operand = np.random.randn(6, 4).astype(np.float64)
    start_indices = np.array([[4], [1], [2]], dtype=np.int64)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
    slice_sizes = [1, 4]
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

    # Input 3
    operand = np.random.randint(-50, 50, size=(3, 3)).astype(np.int32)
    start_indices = np.array([[2], [0]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
    slice_sizes = [1, 3]
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

    # Input 4
    operand = np.random.randn(4, 4).astype(np.float32)
    start_indices = np.array([[2], [1]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
    slice_sizes = [1, 4]
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

    # Input 5
    operand = np.random.choice([True, False], size=(3, 6)).astype(np.bool_)
    start_indices = np.array([[0], [2]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
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

    # Input 6
    operand = np.random.randn(6, 3).astype(np.float32)
    start_indices = np.array([[1], [4], [3]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
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

    # Input 7
    operand = (np.random.randn(4, 2) + 1j * np.random.randn(4, 2)).astype(np.complex64)
    start_indices = np.array([[0], [2]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
    slice_sizes = [1, 2]
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

    # Input 8
    operand = np.random.randint(-100, 100, size=(4, 6)).astype(np.int64)
    start_indices = np.array([[1], [3]], dtype=np.int64)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
    slice_sizes = [1, 6]
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

    # Input 9
    operand = np.random.randn(3, 4).astype(np.float32)
    start_indices = np.array([[0], [2]], dtype=np.int32)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
    slice_sizes = [1, 4]
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

    # Input 10
    operand = np.random.randn(6, 6).astype(np.float64)
    start_indices = np.array([[2], [1], [4]], dtype=np.int64)
    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0,))
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
